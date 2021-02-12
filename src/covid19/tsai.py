import random
from collections import deque
from datetime import timedelta

import torch
from fastai.callback.all import SaveModelCallback, CSVLogger, EarlyStoppingCallback
from fastai.distributed import *
from fastai.metrics import mae, AccumMetric, accuracy, _rmse
from pytorch_forecasting.metrics import SMAPE, MAPE
from tsai.data.core import get_ts_dls, TSDatasets, TSDataLoaders, ToNumpyTensor, ToFloat, flatten_check, skm_to_fastai
from tsai.data.external import check_data
from tsai.data.preparation import SlidingWindow, SlidingWindowPanel, df2xy
from tsai.data.preprocessing import TSStandardize
from tsai.data.validation import get_splits, rmse
from tsai.learner import ts_learner, Learner
import numpy as np
import calendar
from tsai.models.InceptionTimePlus import InceptionTimePlus, InceptionTimePlus17x17, InceptionTimePlus47x47, \
    InceptionTimePlus62x62
from tsai.models.TSTPlus import TSTPlus
from tsai.models.TST import TST
from tsai.models.ResNet import ResNet
from tsai.models.FCNPlus import FCNPlus
from tsai.models.ResCNN import ResCNN
from tsai.models.XceptionTimePlus import XceptionTimePlus
from tsai.models.RNNPlus import LSTMPlus
from tsai.models.RNN_FCNPlus import *
from sklearn.preprocessing import StandardScaler, MinMaxScaler, Normalizer
from matplotlib import pyplot as plt

from . import config
from .datasets import RnboGovUa
import pandas as pd
import os


def skm_smape(y_pred, target):
    y_pred, target = flatten_check(y_pred, target)
    loss = 2 * (y_pred - target).abs() / (y_pred.abs() + target.abs() + 1e-8)
    return loss.mean()


def mape(y_pred, target):
    y_pred, target = flatten_check(y_pred, target)
    loss = (y_pred - target).abs() / (target.abs() + 1e-8)
    return loss.mean()


smape = AccumMetric(skm_smape)


def rescale_columns(df_columns, scaler):
    X = df_columns.values.reshape(-1, 1)
    scaler.fit(X)
    X = scaler.transform(X)
    return X.reshape(-1)


def set_seeds():
    random.seed(42)
    np.random.seed(12345)
    torch.manual_seed(1234)
    torch.set_deterministic(True)
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = False


set_seeds()


class MLSTM_FCNPlus_(MLSTM_FCNPlus):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs, shuffle=False)


def test(fit=True, model_class=InceptionTimePlus17x17, window_length=56, horizon=7):
    ds = RnboGovUa()
    data = RnboGovUa().prepare(
        metrics=RnboGovUa.metrics,
        country_filter=['Ukraine']
    )
    df = data.copy()
    # df = df.loc[df['region'] == 'Dnipropetrovska']
    # df['delta_confirmed_norm'] = rescale_columns(df.delta_confirmed, scaler=Normalizer())
    df['confirmed_std'] = rescale_columns(df.confirmed, scaler=StandardScaler())
    scalers_dict = {
        'confirmed_nx': MinMaxScaler(),
        'existing_nx': MinMaxScaler()
    }
    df['confirmed_nx'] = rescale_columns(df.confirmed, scaler=scalers_dict['confirmed_nx'])
    df['existing_nx'] = rescale_columns(df.existing, scaler=scalers_dict['existing_nx'])
    df['delta_confirmed_nx'] = rescale_columns(df.delta_confirmed, scaler=MinMaxScaler())
    df['delta_confirmed_std'] = rescale_columns(df.delta_confirmed, scaler=StandardScaler())
    df['existing_std'] = rescale_columns(df.existing, scaler=StandardScaler())
    df['suspicion_std'] = rescale_columns(df.suspicion, scaler=StandardScaler())
    df['deaths_std'] = rescale_columns(df.deaths, scaler=StandardScaler())
    df['delta_deaths_std'] = rescale_columns(df.delta_deaths, scaler=StandardScaler())
    df['delta_existing_std'] = rescale_columns(df.delta_existing, scaler=StandardScaler())
    df['delta_existing_nx'] = rescale_columns(df.delta_existing, scaler=MinMaxScaler())
    df['delta_confirmed_nx'] = rescale_columns(df.delta_confirmed, scaler=MinMaxScaler())
    # df['confirmed_yst'] = df.confirmed.shift()
    # df['confirmed_diff'] = df['confirmed'] - df['confirmed_yst']
    # regions_count = len(df.region.unique())
    window_length = window_length

    stride = 1

    for idx, day_name in enumerate(calendar.day_name):
        df[day_name] = df['date'].apply(
            lambda x: 1. if x.day_name() == day_name else .0)

    #
    print(df.head(15))
    print(f'Dataframe: {df.shape}')

    # X_3d, y_3d = df2xy(
    #     df,
    #     feat_col='country_region',
    #     data_cols=['suspicion', 'confirmed'],
    #     target_col=['country_region','delta_confirmed'], to3d=True)
    #
    # print(f'X_3d.shape: {X_3d.shape}')
    # print(f'y_3d.shape: {y_3d.shape}')

    # training_cutoff = df["idx"].max() - horizon
    # train_data = df[lambda x: x.idx <= training_cutoff]

    columns_idx = {i: n for i, n in enumerate(df.columns.values)}

    vars = ['confirmed_std', 'existing_std', 'delta_confirmed_std', 'region_cat'] + list(calendar.day_name)
    vars = [columns_idx[k] for k in sorted(columns_idx.keys()) if columns_idx[k] in vars]

    vars_dict = {k: v for v, k in enumerate(vars)}
    target = ['existing_nx'] # ['confirmed_nx']

    # print(dsets[0])
    # b = dls.one_batch()
    # dls.show_batch()

    # plt.show()
    # dls = get_ts_dls(X, y, splits=splits, tfms=tfms, batch_tfms=batch_tfms, bs=)
    model_name = model_class.__name__
    fname = f'{model_name}_window={window_length}_horizon={horizon}'

    if fit:
        training_cutoff = df.idx.max() - horizon
        train_data = df[lambda x: x.idx <= training_cutoff]
        wl = SlidingWindow(
            window_length,
            seq_first=True,
            get_x=vars,
            get_y=target,
            stride=stride,
            horizon=horizon)

        time_steps = len(train_data.idx.unique())
        X_train = []
        y_train = []
        X_valid = []
        y_valid = []
        for region in train_data.region.unique():
            region_data = train_data.loc[train_data['region'] == region]
            assert len(region_data) == time_steps
            X_region, y_region = wl(region_data)
            y_region = y_region.astype('float32')
            X_train.append(X_region[:-1])
            y_train.append(y_region.astype('float32')[:-1])
            X_valid.append(X_region[-1:])
            y_valid.append(y_region[-1:])

        y_valid = np.vstack(y_valid)
        X_valid = np.vstack(X_valid)
        y_train = np.vstack(y_train)
        X_train = np.vstack(X_train)

        X_train = np.vstack([X_train, X_valid])
        y_train = np.vstack([y_train, y_valid])

        # X_train, y_train = wl(train_data)
        # y_train = y_train.astype('float32')
        # [10 *
        splits = get_splits(y_train, valid_size=.5, stratify=False, random_state=23, shuffle=True)
        # validation_steps = len(train_data.region.unique())
        # total_indexes = list(range(y_train.shape[0]))
        # splits = total_indexes[:-validation_steps], total_indexes[-validation_steps:]
        check_data(X_train, y_train, splits)
        tfms = None
        # batch_tfms = TSStandardize(by_sample=True, by_var=True)
        batch_tfms = None
        # tfms  = [None, [ToFloat(), TSForecasting]]
        dsets = TSDatasets(X_train, y_train, tfms=tfms, splits=splits)
        # SlidingWindowPanel
        dls = TSDataLoaders.from_dsets(dsets.train, dsets.valid, bs=[128, 128], batch_tfms=batch_tfms, num_workers=4,
                                       pin_memory=True)

        model = model_class(c_in=dls.vars, c_out=horizon)  # , seq_len=window_length)
        # model = DataParallel(model)
        learn = Learner(
            dls, model, metrics=[
                mae,
                rmse,
                smape,
                mape
            ],
            cbs=[
                # TensorBoardCallback(projector=False, log_dir='train_log', trace_model=False),
                CSVLogger(fname=f'{fname}.csv'),
                SaveModelCallback(fname=fname),
                EarlyStoppingCallback(min_delta=0, patience=200)
            ]
        )
        # learn.reset()

        # from fastai.distributed import *
        # learn.to_parallel()
        # if torch.cuda.is_available():
        with learn.parallel_ctx():
            r = learn.lr_find()
            print(r)
            print(learn.loss_func)
            learn.fit_one_cycle(10000, 1e-3)
        # else:
        #     learn.fit_one_cycle(1000, 1e-3)

        learn.recorder.plot_metrics()

    testing_cutoff = df.idx.max() - window_length - horizon
    test_data = df[lambda x: x.idx > testing_cutoff]

    # wl = SlidingWindow(
    #     window_length,
    #     seq_first=True,
    #     get_x=vars,
    #     get_y=target,
    #     stride=None,
    #     horizon=horizon)
    # x_test, y_true = wl(test_data)
    # print(true_data)

    time_steps = len(test_data.idx.unique())
    X_test = []
    y_true = []
    for region in test_data.region.unique():
        region_data = test_data.loc[test_data['region'] == region]
        assert len(region_data) == time_steps
        region_history_data = region_data[:window_length]
        region_true_data = region_data[window_length:]
        # history_data = test_data[:window_length]
        # print(history_data)
        # true_data = test_data[window_length:]
        #
        # region_history_data = history_data.loc[history_data['region'] == region]
        # region_true_data = true_data.loc[true_data.region == region]

        X_region = np.asarray([region_history_data[vars].values.transpose(1, 0)])
        y_region = np.asarray([region_true_data[target].values.reshape(-1)])

        assert X_region.shape[2] == window_length
        assert y_region.shape[1] == horizon
        X_test.append(X_region)
        y_true.append(y_region.astype('float32'))

    y_true = np.vstack(y_true)
    X_test = np.vstack(X_test)

    check_data(X_test, y_true)

    split = list(range(len(y_true)))

    dls = get_ts_dls(X=X_test, y=y_true, splits=(split, split))

    model = model_class(c_in=len(vars), c_out=horizon)

    learn = Learner(
        dls, model, metrics=[mae, rmse, smape])
    learn.load(fname, with_opt=False)

    inputs, valid_preds, valid_targets = learn.get_preds(ds_idx=1, with_input=True)

    last_date = test_data['date'].max() - timedelta(days=horizon)

    columns = {
        'region': deque(),
        'mape': deque(),
        'rmse': deque()
    }

    target_name = f'predicted_{target[0]}'

    columns_prediction = {
        'date': deque(),
        'region': deque(),
        target_name: deque(),
    }

    for i in range(horizon):
        # columns[f'Day_{i + window_length + 1} AE'] = deque()
        columns[f'Day_{i + window_length + 1} APE'] = deque()

    inverse_predicts = scalers_dict[target[0]].inverse_transform(valid_preds)

    for sample_idx in range(valid_preds.shape[0]):
        input_ = inputs[sample_idx]
        region_cat = input_[vars_dict['region_cat']][0].long()
        region = df.region.cat.categories[region_cat]
        columns['region'].append(region)
        columns['mape'].append(
            mape(valid_targets[sample_idx], valid_preds[sample_idx])
        )
        columns['rmse'].append(_rmse(valid_targets[sample_idx], valid_preds[sample_idx]).mean())
        for day in range(valid_preds.shape[1]):
            row_date = last_date + timedelta(days=day + 1)
            predicted = valid_preds[sample_idx][day]
            target = valid_targets[sample_idx][day]
            # columns[f'Day_{day + window_length + 1} AE'].append((predicted - target).abs())
            columns[f'Day_{day + window_length + 1} APE'].append(mape(predicted, target))

            columns_prediction['date'].append(row_date)
            columns_prediction['region'].append(region)
            columns_prediction[target_name].append(inverse_predicts[sample_idx][day])

    test_df = pd.DataFrame.from_dict(columns)
    # print(test_df)
    # print(test_df.describe())
    test_df = test_df.append(test_df.describe(), ignore_index=False).fillna("")
    test_df.index.name = 'idx'
    print(test_df)

    test_df.to_csv(f'{fname}_test.csv')

    prediction_df = pd.DataFrame.from_dict(columns_prediction)
    print(prediction_df)
    merged = data.merge(prediction_df, on=['date', 'region'], how='left')
    csv_path = os.path.join(config.DATASETS_DIR, f'{ds.__class__.__name__.lower()}.csv')
    merged.to_csv(csv_path)
