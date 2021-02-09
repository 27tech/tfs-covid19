from fastai.metrics import mae, AccumMetric
# from pmdarima.metrics import smape
from tsai.data.core import get_ts_dls, TSDatasets, TSDataLoaders, ToNumpyTensor, ToFloat, flatten_check, skm_to_fastai
from tsai.data.external import check_data
from tsai.data.preparation import SlidingWindow
from tsai.data.preprocessing import TSStandardize
from tsai.data.validation import get_splits, rmse
from tsai.learner import ts_learner, Learner
from tsai.models.InceptionTime import InceptionTime
from tsai.models.InceptionTimePlus import InceptionTimePlus
from tsai.models.TSTPlus import TSTPlus

from .datasets import RnboGovUa

from pytorch_forecasting.metrics import SMAPE

# s = SMAPE()

def skm_smape(y_pred, target):
    y_pred, target = flatten_check(y_pred, target)
    loss = 2 * (y_pred - target).abs() / (y_pred.abs() + target.abs() + 1e-8)
    return loss.mean()

smape = AccumMetric(skm_smape)

def test():
    window_length = 21
    horizon = 7
    df = RnboGovUa().prepare(metrics=RnboGovUa.metrics, country_filter=['Ukraine'])
    print(df.head(5))
    print(f'Dataframe: {df.shape}')
    wl = SlidingWindow(
        window_length,
        seq_first=True, get_x=['region_cat', 'confirmed', 'suspicion'],
        get_y=['delta_confirmed'],
        horizon=horizon)



    X, y = wl(df)

    y = y.astype('float32')
    from sklearn.preprocessing import StandardScaler, MinMaxScaler, Normalizer
    # scaler = MinMaxScaler()
    # scaler = StandardScaler()
    scaler = Normalizer()
    scaler.fit(y)
    y = scaler.transform(y)

    print(f'X: {X.shape}')
    print(f'y: {y.shape}')

    splits = get_splits(y, valid_size=.2, stratify=False, random_state=23, shuffle=True)
    check_data(X, y, splits)
    tfms = None
    batch_tfms = TSStandardize(by_sample=True, by_var=True)
    # tfms  = [None, [ToFloat()]]
    dsets = TSDatasets(X, y, tfms=tfms, splits=splits)
    dls = TSDataLoaders.from_dsets(dsets.train, dsets.valid, bs=64, batch_tfms=batch_tfms,num_workers=0)
    print(dsets[0])
    # dls = get_ts_dls(X, y, splits=splits, tfms=tfms, batch_tfms=batch_tfms, bs=)
    print(f'dls.c: {dls.c}')
    learn = Learner(dls, InceptionTimePlus(c_in=dls.vars, c_out=horizon), metrics=[mae, rmse, smape])
    # learn = ts_learner(dls, InceptionTime, metrics=[mae, rmse], verbose=True)
    learn.fit_one_cycle(50, 1e-3)



