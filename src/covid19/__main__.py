import calendar

import torch
from pytorch_forecasting.data.encoders import MultiNormalizer, TorchNormalizer, EncoderNormalizer
from pytorch_forecasting.metrics import NormalDistributionLoss, QuantileLoss
from pytorch_forecasting.models.temporal_fusion_transformer.tuning import optimize_hyperparameters
import time
from .datasets import OpenWorldDataset, RnboGovUa
from pathlib import Path
from pytorch_lightning.loggers import TensorBoardLogger
from .logging import configure_logging
from pytorch_forecasting.data.timeseries import TimeSeriesDataSet
import pytorch_lightning as pl

from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor

from pytorch_forecasting import TimeSeriesDataSet, TemporalFusionTransformer, GroupNormalizer, Baseline, DeepAR
from pytorch_forecasting.models import TemporalFusionTransformer, NBeats
from pytorch_lightning.callbacks import ModelCheckpoint
from logging import getLogger

import datetime
import pandas as pd
from pytorch_forecasting.data.examples import generate_ar_data

# data = generate_ar_data(seasonality=10.0, timesteps=400, n_series=2)
#
# data["date"] = pd.Timestamp("2020-01-01") + pd.to_timedelta(data.time_idx, "D")
# print(data.head(20))
# exit(1)

def main():
    max_prediction_length = 7
    max_encoder_length = 28

    logger = getLogger(__name__)
    # OpenWorldDataset.download(Path('owid-covid-latest.csv'))
    ds = RnboGovUa('data')
    data = ds.prepare(metrics={'delta_confirmed'})
    # data = data.drop(columns=['country'])
    # print(data.head(10))

    # data = data.groupby(['idx', 'country', 'date']).sum('delta_confirmed').reset_index()
    data.to_csv('dataset.csv')
    print(data.head(10))

    # for idx, day_name in enumerate(calendar.day_name):
    #     data[day_name] = data['date'].apply(
    #         lambda x: day_name if x.day_name() == day_name else "-").astype('category')
    #
    print(data.head(10))
    # exit(1)
    training_cutoff = data["idx"].max() - max_prediction_length
    group_ids = ['country', 'region']
    # group_ids = ['country']

    training = TimeSeriesDataSet(
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
            groups=group_ids, #, transformation="softplus"
        ),
            # groups=groups, transformation="softplus"
        # ),  # use softplus and normalize by group
        # add_relative_time_idx=True,
        # add_target_scales=True,
        # add_encoder_length=True,
        # add_relative_time_idx=True,
          # add_target_scales=True,  # add as feature
          # add_encoder_length=True,
        allow_missings=True
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

    learning_rate = 0.03
    gradient_clip_val = 0.01
    batch_size = 32  # set this between 32 to 128
    weight_decay = 0.001

    # create validation set (predict=True) which means to predict the last max_prediction_length points in time
    # for each series
    validation = TimeSeriesDataSet.from_dataset(training, data, predict=True, stop_randomization=True)

    # create dataloaders for model
    train_dataloader = training.to_dataloader(train=True, batch_size=batch_size, num_workers=8)
    val_dataloader = validation.to_dataloader(train=False, batch_size=batch_size, num_workers=8)

    early_stop_callback = EarlyStopping(monitor="val_loss", min_delta=1e-4, patience=100, verbose=True, mode="min")
    # early_stop_callback = EarlyStopping(monitor="val_MAE", min_delta=1e-1, patience=100, verbose=True, mode="max")
    checkpoint_callback = ModelCheckpoint(monitor='val_loss', verbose=True)

    lr_logger = LearningRateMonitor()

    # configure network and trainer
    pl.seed_everything(42)


    if 0:
        net = TemporalFusionTransformer.from_dataset(
            training,
            learning_rate=learning_rate,
            # hidden_size=16,
            # attention_head_size=1,
            dropout=0.1,
            # hidden_continuous_size=8,
            # output_size=7,  # 7 quantiles by default
            # loss=QuantileLoss(),
            # log_interval=10,  # uncomment for learning rate finder and otherwise, e.g. to 10 for logging every 10 batches
            # reduce_on_plateau_patience=4,
        )
        #print(f"Number of parameters in network: {tft.size() / 1e3:.1f}k")
    if 0:
        net = NBeats.from_dataset(
            training,
            # stack_types = ['generic'],
            # num_blocks=[4, 4],
            # num_block_layers=[4, 4],
            learning_rate=learning_rate,
            # log_interval=10,
            # log_val_interval=1,
            log_gradient_flow=False,
            weight_decay=weight_decay,
            # reduce_on_plateau_patience=50
        )

    if 1:
        net = DeepAR.from_dataset(
            training,
            # learning_rate=3e-2,
            learning_rate=learning_rate,
            hidden_size=8,
            dropout=0.1,
            loss=NormalDistributionLoss(),
            # log_interval=1.0,
            # log_val_interval=100,
            log_gradient_flow=False,
            weight_decay=1e-2,
            # optimizer='adamw'
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
            gpus=[1] if torch.cuda.is_available() else 0,
            weights_summary="top",
            gradient_clip_val=gradient_clip_val,
            limit_train_batches=100.,  # coment in for training, running valiation every 30 batches
            # fast_dev_run=True,  # comment in to check that networkor dataset has no serious bugs
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
            default_root_dir='checkpoints'
        )

        # fit network
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
    predictions = net.predict(val_dataloader)
    print(f"Mean absolute error of model: {(actuals - predictions).abs().mean()}")

    # actuals = torch.cat([y for x, (y, weight) in iter(val_dataloader)])
    # baseline_predictions = Baseline().predict(val_dataloader)
    # print((actuals - baseline_predictions).abs().mean().item())

    # ds.download()

if __name__ == "__main__":
    configure_logging()
    main()
