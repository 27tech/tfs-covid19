# import random
import calendar
import os
from collections import deque
from datetime import timedelta
from fastai.distributed import *

# noinspection PyPackageRequirements
import numpy as np
import pandas as pd
# import torch
from fastai.callback.all import SaveModelCallback, CSVLogger, EarlyStoppingCallback
# from tsai.data.preprocessing import TSStandardize
from fastai.learner import *
from tsai.data.preprocessing import TSStandardize, TSNormalize
from tsai.learner import *
from fastai.metrics import mae, rmse, mse
# InceptionTimePlus, InceptionTimePlus17x17, InceptionTimePlus47x47, \
#     InceptionTimePlus62x62
# from tsai.models.TSTPlus import TSTPlus
# from tsai.models.TST import TST
from tsai.models.ResNet import ResNet
from tsai.models.FCNPlus import FCNPlus
# from tsai.models.ResCNN import ResCNN
# from tsai.models.XceptionTimePlus import XceptionTimePlus
# from tsai.models.RNNPlus import LSTMPlus
# from tsai.models.RNN_FCNPlus import *
# from sklearn.preprocessing import StandardScaler, MinMaxScaler, Normalizer
from tsai.data.core import get_ts_dls, TSDatasets, TSDataLoaders  # , ToNumpyTensor, ToFloat
from tsai.data.external import check_data
# from tsai.data.preparation import SlidingWindow
from tsai.models.InceptionTimePlus import InceptionTimePlus17x17

from . import config
from .datasets import Rnbo
from .metrics import mape, smape, rmse, mpe
from .seed import set_seeds


# noinspection PyProtectedMember


# def rescale_columns(df_columns, scaler):
#     X = df_columns.values.reshape(-1, 1)
#     scaler.fit(X)
#     X = scaler.transform(X)
#     return X.reshape(-1)


def test(fit=True, model_class=InceptionTimePlus17x17, window_length=56, horizon=14):
    # set_seeds()
    ds = Rnbo()
    # ds.get
    data = RnboGovUa().prepare(
        metrics=RnboGovUa.metrics,
        country_filter=[
            'Ukraine',
        #     'China', 'Thailand', 'Singapore', 'Japan',
        #     'Korea, South', 'Australia', 'Germany', 'US',
        #     'Taiwan*', 'Malaysia', 'Vietnam'
        ]
    )
    # data = data.loc[~data.country.isin([
    #     'Belgium', 'Malawi', 'Western Sahara', 'South Sudan',
    #     'Sao Tome and Principe', 'Yemen'
    # ])]
    # print(f"countries: {data.country.unique()}")
    df = data.copy()
    df = df.loc[df['region'] == 'Kyiv']
    # df['delta_confirmed_norm'] = rescale_columns(df.delta_confirmed, scaler=Normalizer())
    # df['confirmed_std'] = rescale_columns(df.confirmed, scaler=StandardScaler())
    # scalers_dict = {
    #     'confirmed_nx': MinMaxScaler(),
    #     'existing_nx': MinMaxScaler(),
    #     'existing_std': StandardScaler(),
    #     'delta_confirmed_nx': MinMaxScaler(),
    #     'delta_existing_nx': MinMaxScaler(),
    # }
    # df['confirmed_nx'] = rescale_columns(df.confirmed, scaler=scalers_dict['confirmed_nx'])
    # df['existing_nx'] = rescale_columns(df.existing, scaler=scalers_dict['existing_nx'])
    # df['delta_confirmed_nx'] = rescale_columns(df.delta_confirmed, scaler=MinMaxScaler())
    # df['delta_confirmed_std'] = rescale_columns(df.delta_confirmed, scaler=StandardScaler())
    # df['existing_std'] = rescale_columns(df.existing, scaler=scalers_dict['existing_std'])
    # df['suspicion_std'] = rescale_columns(df.suspicion, scaler=StandardScaler())
    # df['deaths_std'] = rescale_columns(df.deaths, scaler=StandardScaler())
    # df['delta_deaths_std'] = rescale_columns(df.delta_deaths, scaler=StandardScaler())
    # df['delta_deaths_nx'] = rescale_columns(df.delta_deaths, scaler=MinMaxScaler())
    # df['delta_existing_std'] = rescale_columns(df.delta_existing, scaler=StandardScaler())
    # df['delta_existing_nx'] = rescale_columns(df.delta_existing, scaler=scalers_dict['delta_existing_nx'])
    # df['delta_confirmed_nx'] = rescale_columns(df.delta_confirmed, scaler=scalers_dict['delta_confirmed_nx'])
    # df['lat_nx'] = rescale_columns(df.lat, scaler=MinMaxScaler())
    # df['lng_nx'] = rescale_columns(df.lng, scaler=MinMaxScaler())
    # df['confirmed_yst'] = df.confirmed.shift()
    # df['confirmed_diff'] = df['confirmed'] - df['confirmed_yst']
    # regions_count = len(df.region.unique())
    window_length = window_length

    stride = 1


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

    group_name = 'country_region_cat'
    group_category = 'country_region'

    # featured_columns = ['confirmed', 'existing', 'delta_confirmed']

    # scalers_dict['existing_norm'] = StandardScaler()

    # scalers_dict = normalize(df, RnboGovUa.metrics)

    features = [
            'existing_nx',
            'delta_existing_nx',
            # 'delta_confirmed_nx',
            # 'delta_suspicion_nx',
            # 'delta_recovered_nx',
            # 'delta_deaths_nx',
                   # 'confirmed_nx',
            # 'delta_existing_rob',
            # 'delta_existing_nx',
            # 'delta_existing_std',

            # 'deaths_nx',
            # 'recovered_std',
            # 'existing',
            # 'suspicion_nx',

            # 'delta_existing_nx',
            # group_name
    ] + list(calendar.day_name)

    print(f'Features: {features}')
    columns_idx = {i: n for i, n in enumerate(df.columns.values)}

    features = [columns_idx[k] for k in sorted(columns_idx.keys()) if columns_idx[k] in features]

    vars_dict = {k: v for v, k in enumerate(features)}
    target = ['delta_existing_std']  # ['confirmed_nx']
    print(df[features + target].sample(5))
    model_name = model_class.__name__
    fname = f'{model_name}_window={window_length}_horizon={horizon}_{"-".join(features)}'

    if fit:
        training_cutoff = df.idx.max() - horizon
        train_data = df[df.idx <= training_cutoff]
        # train_data = df
        print('Train Data Tail')
        print(train_data.tail(5))
        wl = SlidingWindow(
            window_length,
            seq_first=True,
            get_x=features,
            get_y=target,
            stride=stride,
            horizon=horizon)

        time_steps = len(train_data.idx.unique())
        X_train = []
        y_train = []
        X_valid = []
        y_valid = []
        back_steps = -1
        print(f'back_steps: {back_steps}')
        for region in train_data[group_name].unique():
            region_data = train_data.loc[train_data[group_name] == region]
            if len(region_data) != time_steps:
                print(f'Skip: {df[group_category].cat.categories[region]}')
                continue
            assert len(region_data) == time_steps, f'Region {df[group_category].cat.categories[region]} != {time_steps}'
            X_region, y_region = wl(region_data)
            y_region = y_region.astype('float32')
            X_train.append(X_region[:back_steps])
            y_train.append(y_region.astype('float32')[:back_steps])
            X_valid.append(X_region[back_steps:])
            y_valid.append(y_region[back_steps:])

        y_valid = np.vstack(y_valid)
        X_valid = np.vstack(X_valid)
        y_train = np.vstack(y_train)
        X_train = np.vstack(X_train)

        X_train = np.vstack([X_train, X_valid])
        y_train = np.vstack([y_train, y_valid])

        # X_train, y_train = wl(train_data)
        # y_train = y_train.astype('float32')
        # [10 *
        # splits = get_splits(y_train, valid_size=.5, stratify=False, random_state=23, shuffle=True)
        validation_steps = len(y_valid)
        total_indexes = list(range(y_train.shape[0]))
        splits = total_indexes[:-validation_steps], total_indexes[-validation_steps:]
        check_data(X_train, y_train, splits)
        tfms = None
        # batch_tfms = TSStandardize(by_sample=True, by_var=True)
        batch_tfms = None
        tfms = None # [None, [TSNormalize()]]
        dsets = TSDatasets(X_train, y_train, tfms=tfms, splits=splits)
        dls = TSDataLoaders.from_dsets(dsets.train, dsets.valid, bs=[32, 128], batch_tfms=batch_tfms, num_workers=0,
                                       pin_memory=True)

        model = model_class(c_in=dls.vars, c_out=horizon)  # , seq_len=window_length)
        # model = DataParallel(model)
        learn = Learner(
            dls, model, metrics=[
                mse,
                mae,
                rmse,
                smape,
                mape
            ],
            cbs=[
                # TensorBoardCallback(projector=False, log_dir='train_log', trace_model=False),
                CSVLogger(fname=f'{fname}.csv'),
                SaveModelCallback(fname=fname, with_opt=True),
                EarlyStoppingCallback(min_delta=0, patience=50)
            ]
        )
        # learn.reset()

        # from fastai.distributed import *
        # learn.to_parallel()
        # if torch.cuda.is_available():
        with learn.parallel_ctx():
            # learn.fine_tune(10)
            # print(r)
            # print(learn.loss_func)
            r = learn.lr_find()
            print(r)
            learn.fit_one_cycle(100000, 1e-3)
        # else:
        #     learn.fit_one_cycle(1000, 1e-3)

        # learn.recorder.plot_metrics()

    testing_cutoff = df.idx.max() - window_length - horizon
    test_data = df[lambda x: x.idx > testing_cutoff]


    time_steps = len(test_data.idx.unique())
    X_test = []
    y_true = []
    for region in test_data[group_name].unique():
        region_data = test_data.loc[test_data[group_name] == region]
        if len(region_data) != time_steps:
            print(f'Skip: {df[group_category].cat.categories[region]}')
            continue
        assert len(region_data) == time_steps
        region_history_data = region_data[:window_length]
        region_true_data = region_data[window_length:]
        # history_data = test_data[:window_length]
        # print(history_data)
        # true_data = test_data[window_length:]
        #
        # region_history_data = history_data.loc[history_data['region'] == region]
        # region_true_data = true_data.loc[true_data.region == region]

        X_region = np.asarray([region_history_data[features].values.transpose(1, 0).astype('float32')])
        y_region = np.asarray([region_true_data[target].values.reshape(-1).astype('float32')])

        assert X_region.shape[2] == window_length
        assert y_region.shape[1] == horizon
        X_test.append(X_region.astype('float32'))
        y_true.append(y_region.astype('float32'))

    y_true = np.vstack(y_true)
    X_test = np.vstack(X_test)

    check_data(X_test, y_true)

    split = list(range(len(y_true)))

    dls = get_ts_dls(X=X_test, y=y_true, splits=(split, split))

    model = model_class(c_in=len(features), c_out=horizon)

    learn = Learner(
        dls, model, metrics=[mae, rmse, smape])
    learn.load(fname, with_opt=False)

    inputs, valid_preds, valid_targets = learn.get_preds(ds_idx=1, with_input=True)

    last_date = test_data['date'].max() - timedelta(days=horizon)

    columns = {
        group_category: deque(),
        'mape': deque(),
        'rmse': deque()
    }

    target_name = f'predicted_{target[0]}'

    columns_prediction = {
        'date': deque(),
        group_category: deque(),
        target_name: deque(),
    }

    for i in range(horizon):
        columns[f'Day_{i + window_length + 1} MPE'] = deque()
        columns[f'Day_{i + window_length + 1} MAPE'] = deque()

    inverse_predicts = scalers_dict[target[0]].inverse_transform(valid_preds)
    errors_df = pd.DataFrame(
        {
            'Test MSE': [mse(valid_targets, valid_preds)],
            'Test MAE': [mae(valid_targets, valid_preds).mean()],
            'Test RMSE': [rmse(valid_targets, valid_preds).mean()],
            'Test SMAPE': [smape(valid_targets, valid_preds).mean()],
            'Test MAPE': [mape(valid_targets, valid_preds).mean()]
        }
    )

    print(errors_df)

    for sample_idx in range(valid_preds.shape[0]):
        input_ = inputs[sample_idx]
        region_cat = input_[vars_dict[group_name]][0].long()
        region = df[group_category].cat.categories[region_cat]
        columns[group_category].append(region)
        columns['mape'].append(
            mape(valid_targets[sample_idx], valid_preds[sample_idx])
        )
        columns['rmse'].append(rmse(valid_targets[sample_idx], valid_preds[sample_idx]))
        for day in range(valid_preds.shape[1]):
            row_date = last_date + timedelta(days=day + 1)
            predicted = valid_preds[sample_idx][day]
            target = valid_targets[sample_idx][day]
            # columns[f'Day_{day + window_length + 1} AE'].append((predicted - target).abs())
            columns[f'Day_{day + window_length + 1} MPE'].append(mpe(predicted, target))
            columns[f'Day_{day + window_length + 1} MAPE'].append(mape(predicted, target))

            columns_prediction['date'].append(row_date)
            columns_prediction[group_category].append(region)
            columns_prediction[target_name].append(inverse_predicts[sample_idx][day])

    test_df = pd.DataFrame.from_dict(columns)

    test_df = test_df.append(test_df.describe(), ignore_index=False).fillna("")
    test_df.index.name = 'idx'
    print(test_df)

    test_df.to_csv(f'{fname}_test.csv')

    prediction_df = pd.DataFrame.from_dict(columns_prediction)
    print(prediction_df)
    merged = data.merge(prediction_df, on=['date', group_category], how='left')
    csv_path = os.path.join(config.DATASETS_DIR, f'{ds.__class__.__name__.lower()}.csv')
    merged.to_csv(csv_path)
