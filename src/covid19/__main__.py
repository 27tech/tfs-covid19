import calendar
from typing import Optional
from collections import deque
import torch
from pytorch_forecasting.data.encoders import MultiNormalizer, TorchNormalizer, EncoderNormalizer
from pytorch_forecasting.metrics import NormalDistributionLoss, QuantileLoss
from pytorch_forecasting.models.temporal_fusion_transformer.tuning import optimize_hyperparameters
import time
from .datasets import OpenWorldDataset, RnboGovUa
from pathlib import Path
from pytorch_lightning.loggers import TensorBoardLogger
# from .logging import configure_logging
from pytorch_forecasting.data.timeseries import TimeSeriesDataSet
import pytorch_lightning as pl

from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor
from pytorch_forecasting.metrics import SMAPE

from pytorch_forecasting import TimeSeriesDataSet, TemporalFusionTransformer, GroupNormalizer, Baseline, DeepAR
from pytorch_forecasting.models import TemporalFusionTransformer, NBeats
from pytorch_lightning.callbacks import ModelCheckpoint
from logging import getLogger
from . import config
import os
import datetime
import pandas as pd
from pytorch_forecasting.data.examples import generate_ar_data


# datasets = generate_ar_data(seasonality=10.0, timesteps=400, n_series=2)
#
# datasets["date"] = pd.Timestamp("2020-01-01") + pd.to_timedelta(datasets.time_idx, "D")
# print(datasets.head(20))
# exit(1)

def main():
    max_prediction_length = 7
    max_encoder_length = 21

    logger = getLogger(__name__)
    # OpenWorldDataset.download(Path('owid-covid-latest.csv'))

    ds = RnboGovUa()
    # data = ds.prepare_numpy(x_metrics=RnboGovUa.metrics, y_metrics={'delta_confirmed'}, country_filter=['Ukraine'])
    # array = data.to_numpy()
    data = ds.prepare(metrics=RnboGovUa.metrics, country_filter=['Ukraine'])
    csv_path = os.path.join(config.DATASETS_DIR, f'{ds.__class__.__name__.lower()}.csv')
    # logger.info(f'Save CSV to {csv_path}')
    # data.to_csv(csv_path)
    # datasets = datasets.drop(columns=['country'])
    # print(datasets.head(10))

    # datasets = datasets.groupby(['idx', 'country', 'date']).sum('delta_confirmed').reset_index()
    # datasets.to_csv('dataset.csv')
    # print(datasets.head(10))

    # for idx, day_name in enumerate(calendar.day_name):
    #     data[day_name] = data['date'].apply(
    #         lambda x: day_name if x.day_name() == day_name else "-").astype('category')

    # print(datasets.head(10))

    # exit(1)

    group_ids = ['country', 'region']
    # group_ids = ['region']
    # group_ids = ['region', 'weekday']
    data['weekday'] = data['date'].dt.day_name()
    # print(data[:,'date', 'time_idx'].head(30))
    training_cutoff = data["idx"].max() - max_prediction_length
    train_data = data[lambda x: x.idx <= training_cutoff]

    # target = 'delta_deaths'
    target = 'delta_confirmed'
    training = TimeSeriesDataSet(
        train_data,
        time_idx="idx",
        target=target,
        group_ids=group_ids,
        # min_encoder_length=max_encoder_length // 2,  # keep encoder length long (as it is in the validation set)
        max_encoder_length=max_encoder_length,
        # min_prediction_length=1,
        max_prediction_length=max_prediction_length,
        # static_categoricals=group_ids,
        # static_reals=['weekday'],
        # time_varying_known_categoricals=["weekday"],
        # variable_groups={"day_name": list(calendar.day_name)},  # group of categorical variables can be treated as one variable
        # time_varying_known_categoricals=['day_name'],
        # time_varying_known_reals=['existing'],
        # time_varying_unknown_categoricals=["weekday"],
        time_varying_unknown_reals=[
            # 'existing',
            target
        ],
        # [
        #     "volume",
        #     "log_volume",
        #     "industry_volume",
        #     "soda_volume",
        #     "avg_max_temp",
        #     "avg_volume_by_agency",
        #     "avg_volume_by_sku",
        # ],
        randomize_length=None,
        target_normalizer=GroupNormalizer(
            groups=group_ids
        ),
        # groups=groups, transformation="softplus"
        # ),  # use softplus and normalize by group
        # add_relative_time_idx=True,
        # add_target_scales=True,
        # add_encoder_length=True,
        # add_relative_time_idx=True,
        # add_target_scales=True,  # add as feature
        # add_encoder_length=True,
        # allow_missings=True
    )

    # # convert the dataset to a dataloader
    # dataloader = training.to_dataloader(batch_size=4)
    #
    # # and load the first batch
    # x, y = next(iter(dataloader))
    # print("x =", x)
    # print("\ny =", y)
    # print("\nsizes of x =")
    # for key, value in x.items():
    #     print(f"\t{key} = {value.size()}")

    learning_rate = 1e-1
    gradient_clip_val = 0.1
    batch_size = 8  # set this between 32 to 128
    weight_decay = 0.001

    # create validation set (predict=True) which means to predict the last max_prediction_length points in time
    # for each series
    validation = TimeSeriesDataSet.from_dataset(training, data, predict=True, stop_randomization=True)
    print(len(training))
    # create dataloaders for model
    train_dataloader = training.to_dataloader(train=True, batch_size=batch_size, num_workers=2)
    val_dataloader = validation.to_dataloader(train=False, batch_size=batch_size, num_workers=2)

    early_stop_patience = 5
    reduce_on_plateau_patience = early_stop_patience // 4
    early_stop_callback = EarlyStopping(monitor="val_loss", min_delta=1e-4, patience=early_stop_patience, verbose=True,
                                        mode="min")
    # early_stop_callback = EarlyStopping(monitor="val_MAE", min_delta=1e-1, patience=100, verbose=True, mode="max")
    checkpoint_callback = ModelCheckpoint(monitor='val_loss', verbose=True)

    lr_logger = LearningRateMonitor()

    # configure network and trainer
    pl.seed_everything(42)

    if 0:
        net = TemporalFusionTransformer.from_dataset(
            training,
            # learning_rate=learning_rate,
            # hidden_size=16,
            attention_head_size=1,
            dropout=0.1,
            # hidden_continuous_size=8,
            # output_size=7,  # 7 quantiles by default
            # loss=QuantileLoss(),
            # log_interval=10,  # uncomment for learning rate finder and otherwise, e.g. to 10 for logging every 10 batches
            reduce_on_plateau_patience=reduce_on_plateau_patience,
            weight_decay=weight_decay,
        )
        # print(f"Number of parameters in network: {tft.size() / 1e3:.1f}k")
    if 1:
        net = NBeats.from_dataset(
            training,
            # stack_types = ['generic', 'trend', ],
            # num_blocks=[3, 3],
            # num_block_layers=[5, 5],
            learning_rate=1.0e-4,
            # gradient_clip_val=gradient_clip_val,
            # learning_rate=0.003981071705534969,
            # learning_rate=7.585775750291837e-08,
            # log_interval=10,
            # log_val_interval=1,
            # log_gradient_flow=False,
            weight_decay=weight_decay,
            reduce_on_plateau_patience=reduce_on_plateau_patience
        )

    if 0:
        net = DeepAR.from_dataset(
            training,
            # learning_rate=3e-2,
            learning_rate=learning_rate,
            # hidden_size=32,
            # rnn_layers=4,
            dropout=0.1,
            loss=NormalDistributionLoss(),
            log_interval=100,
            # log_val_interval=100,
            log_gradient_flow=False,
            weight_decay=weight_decay,
            # optimizer='adam',
            reduce_on_plateau_patience=reduce_on_plateau_patience
        )

    print(f"Number of parameters in network: {net.size() / 1e3:.1f}k")

    # find optimal learning rate
    # res = trainer.tuner.lr_find(
    #     deepar,
    #     train_dataloader=train_dataloader,
    #     val_dataloaders=val_dataloader,
    #     max_lr=10.0,
    #     min_lr=1e-6,
    # )

    # print(f"suggested learning rate: {res.suggestion()}")
    # fig = res.plot(show=True, suggest=True)
    # fig.show()

    torch.set_num_threads(10)

    if 1:
        trainer = pl.Trainer(
            # accelerator="dpp",
            max_epochs=1000,
            gpus=[0] if torch.cuda.is_available() else None,
            weights_summary="top",
            gradient_clip_val=gradient_clip_val,
            # limit_train_batches=100.,  # coment in for training, running valiation every 30 batches
            # fast_dev_run=True,  # comment in to check that networkor dataset has no serious bugs
            log_every_n_steps=1000,
            flush_logs_every_n_steps=1000,
            callbacks=[
                lr_logger,
                early_stop_callback,
                checkpoint_callback
            ],
            logger=TensorBoardLogger(
                save_dir='train_log',
                name=net.__class__.__name__.lower(),
                version=f'{int(time.time())}-gcv={gradient_clip_val:.2f}-'
                        f'lr={learning_rate:.3f}-bs={batch_size}-wd={weight_decay:.3f}-'
                        f'win={max_encoder_length}_{max_prediction_length}'
            ),
            default_root_dir='checkpoints',
            # benchmark=True,
            # deterministic=True,
            # profiler=True,
            # enable_pl_optimizer=True
            # auto_lr_find=True
            # auto_scale_batch_size=True
        )

        # # Run learning rate finder
        # lr_finder = trainer.tuner.lr_find(
        #     net, train_dataloader, val_dataloader, num_training=1000)
        #
        # # Results can be found in
        # lr_finder.results
        #
        # # Plot with
        # fig = lr_finder.plot(suggest=True)
        # fig.show()
        #
        # # Pick point based on plot, or get suggestion
        # new_lr = lr_finder.suggestion()
        #
        # # update hparams of the model
        # net.hparams.lr = new_lr

        # print('tune')
        # r = trainer.tune(net, train_dataloader, val_dataloader)
        # print('fit')
        # # fit network
        trainer.fit(
            net,
            train_dataloader=train_dataloader,
            val_dataloaders=val_dataloader,
        )

        print(f'Load checkpoint {checkpoint_callback.best_model_path}')

        net = net.load_from_checkpoint(checkpoint_callback.best_model_path)

    # net.save_hyperparameters()
    # create study
    # study = optimize_hyperparameters(
    #     train_dataloader,
    #     val_dataloader,
    #     model_path=checkpoint_callback.best_model_path,
    #     n_trials=200,
    #     max_epochs=50,
    #     gradient_clip_val_range=(0.01, 1.0),
    #     hidden_size_range=(8, 128),
    #     hidden_continuous_size_range=(8, 128),
    #     attention_head_size_range=(1, 4),
    #     learning_rate_range=(0.001, 0.1),
    #     dropout_range=(0.1, 0.3),
    #     trainer_kwargs=dict(limit_train_batches=30),
    #     reduce_on_plateau_patience=10,
    #     use_learning_rate_finder=False  # use Optuna to find ideal learning rate or use in-built learning rate finder
    # )

    #

    # show best hyperparameters
    # print(study.best_trial.params)

    # calculate mean absolute error on validation set
    actuals = torch.cat([y for x, (y, weight) in iter(val_dataloader)])
    output, x, index, decoder_lengths = net.predict(val_dataloader, return_index=True, return_decoder_lengths=True,
                                                    return_x=True)

    # raw, x, index, decoder_lengths = net.predict(val_dataloader, return_index=True, return_decoder_lengths=True,
    #                                                 return_x=True, mode=('raw', 'prediction'))

    # plot = net.plot_prediction(x=x, out=dict(prediction=raw), idx=0)
    # plot.show()

    target_name = f'predicted_{validation.target}'
    columns = {
        validation.time_idx: deque(),
        target_name: deque(),
        'date': deque()
    }
    for group in validation.group_ids:
        columns[group] = deque()
    last_date = train_data['date'].max()
    for category_idx in range(output.size(0)):
        for time_idx in range(output.size(1)):
            columns[validation.time_idx].append(index.loc[category_idx].idx + output.size(1))
            columns['date'].append(last_date + datetime.timedelta(days=time_idx + 1))
            columns[target_name].append(output[category_idx][time_idx].long().cpu())
            for group in validation.group_ids:
                columns[group].append(index.loc[category_idx][group])

    df = pd.DataFrame.from_dict(columns)
    print(df)
    merged = data.merge(df, on=['date', 'country', 'region'], how='left')
    merged.to_csv(csv_path)
    # print([index.loc[idx, ['region', 'idx']]['region'] for idx in range(output.size(0))])
    # predictions = net.predict(val_dataloader)
    print(f"Mean absolute error of model: {(actuals - output).abs().mean()}")
    # print(f'SMAPE: {SMAPE().loss(actuals, output)}')

    # actuals = torch.cat([y for x, (y, weight) in iter(val_dataloader)])
    # baseline_predictions = Baseline().predict(val_dataloader)
    # print((actuals - baseline_predictions).abs().mean().item())

    # ds.download()


def get_predict_dataset(max_encoder_length: int = 28, max_prediction_length: int = 7):
    rnbo = RnboGovUa()
    data = rnbo.prepare(metrics=rnbo.metrics, country_filter=['Ukraine'])

    training_cutoff = data["idx"].max() - max_encoder_length

    group_ids = ['country', 'region']
    group_ids = ['region']

    dataset = TimeSeriesDataSet(
        data[lambda x: x.idx <= training_cutoff],
        time_idx="idx",
        target='delta_confirmed',
        group_ids=group_ids,
        # min_encoder_length=max_encoder_length // 2,  # keep encoder length long (as it is in the validation set)
        max_encoder_length=max_encoder_length,
        # min_prediction_length=1,
        max_prediction_length=max_prediction_length,
        static_categoricals=group_ids,
        # static_reals=groups,
        # time_varying_known_categoricals=["special_days", "month"],
        # variable_groups={"day_name": list(calendar.day_name)},  # group of categorical variables can be treated as one variable
        # time_varying_known_categoricals=['day_name'],
        # time_varying_known_reals=list(calendar.day_name),
        # time_varying_unknown_categoricals=[],
        time_varying_unknown_reals=[
            # 'existing',
            'delta_confirmed'
        ],
        # [
        #     "volume",
        #     "log_volume",
        #     "industry_volume",
        #     "soda_volume",
        #     "avg_max_temp",
        #     "avg_volume_by_agency",
        #     "avg_volume_by_sku",
        # ],
        randomize_length=None,
        target_normalizer=GroupNormalizer(
            groups=group_ids
        ),
        # groups=groups, transformation="softplus"
        # ),  # use softplus and normalize by group
        # add_relative_time_idx=True,
        add_target_scales=True,
        # add_encoder_length=True,
        # add_relative_time_idx=True,
        # add_target_scales=True,  # add as feature
        # add_encoder_length=True,
        # allow_missings=True
    )

    # assert len(dataset) == max_encoder_length
    return dataset, data


def predict(model_class, checkpoint_path: Optional[str]):
    if checkpoint_path is None:
        checkpoint_path = os.path.join(config.CHECKPOINTS_DIR, f'{model_class.__name__.lower()}.ckpt')

    # create validation set (predict=True) which means to predict the last max_prediction_length points in time
    # for each series
    training, data = get_predict_dataset()
    validation = TimeSeriesDataSet.from_dataset(training, data, predict=True, stop_randomization=True)
    print(len(validation))
    # create dataloaders for model
    # train_dataloader = training.to_dataloader(train=True, batch_size=batch_size, num_workers=8)
    dataloader = validation.to_dataloader(train=False, batch_size=1, num_workers=0)

    net = model_class.load_from_checkpoint(checkpoint_path)
    # actuals = torch.cat([y for x, (y, weight) in iter(dataloader)])
    predictions = net.predict(dataloader)
    print(predictions)
    # print(f"Mean absolute error of model: {(actuals - predictions).abs().mean()}")


from .tsai import test

if __name__ == "__main__":
    # predict(model_class=DeepAR, checkpoint_path=None)
    test()
    # main()
