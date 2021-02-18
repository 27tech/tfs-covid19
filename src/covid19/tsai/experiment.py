import json
from datetime import datetime, timedelta
from logging import getLogger
from typing import List, Tuple, Optional, Dict, Any
from pandas import DataFrame

import numpy as np
from torch import Tensor
from tsai.data.core import TSDatasets, TSDataLoaders
from tsai.data.external import check_data
from tsai.learner import ts_learner
from tsai.models.InceptionTime import InceptionTime
from tsai.models.InceptionTimePlus import InceptionTimePlus

from covid19.datasets.rnbo import Rnbo
from .learner import TSAILearner
import os
from covid19 import config

logger = getLogger(__name__)


class Experiment:
    def __init__(self, model_class: type, lr: float, early_stop_patience: int, epochs: int, features: List[str],
                 targets: List[str], window: int, horizon: int, batch_size: int, country_filter: List[str],
                 region_filter: Optional[List[str]] = None, group_name='country_region'):
        self._lr: float = lr
        self._epochs: int = epochs
        self._name: str = datetime.now().strftime(f'%Y-%m-%d %H:%M {model_class.__name__}')
        self._group_name = group_name
        self._group_name_cat = f'{group_name}_cat'
        self._features = features + [self._group_name_cat]
        self._targets = targets
        self._window = window
        self._horizon = horizon
        self._batch_size = batch_size
        self._model_class = model_class
        self._early_stop_patience = early_stop_patience
        self._country_filter = country_filter
        self._region_filter = region_filter
        self._dir = os.path.join(config.CHECKPOINTS_DIR, self._name)

        os.makedirs(self._dir, exist_ok=True)

        assert len(self._targets) == 1

        self._dataset = Rnbo()

        self._dataset.scrape(
            country_filter=self._country_filter, metrics=Rnbo.metrics, region_filter=self._region_filter)

        population = 328200000.0

        self._dataset._dataframe['confirmed_pop'] = self._dataset._dataframe['confirmed'] / population
        self._dataset._dataframe['existing_pop'] = self._dataset._dataframe['existing'] / population
        self._dataset._dataframe['none_sick_pop'] = 1. - self._dataset._dataframe['confirmed_pop']


        self._columns_vocab = {i: n for i, n in enumerate(self._dataset.data_frame.columns.values)}

        self._features_idx = [
            self._columns_vocab[k] for k in sorted(self._columns_vocab.keys()) if
            self._columns_vocab[k] in self._features]

        self._features_vocab = {k: v for v, k in enumerate(self._features_idx)}

        self._group_name_cat_idx = self._features_vocab[self._group_name_cat]

        self._target_name = "_".join(self._targets[0].split('_')[0:-1])

        self._target_name_predict = f'{self._target_name}_predict'

    def _construct_model(self, dls: TSDataLoaders):
        # ts_learner(dls, InceptionTimePlus, metrics=[], cbs=None)
        # return self._model_class(c_in=dls.vars, seq_len=dls.len, c_out=dls.c)
        return self._model_class(c_in=len(self._features), c_out=self._horizon, seq_len=self._window)

    def get_dls(self, data: Tuple[np.ndarray, np.ndarray], splits: Tuple[np.ndarray, np.ndarray]) -> TSDataLoaders:
        x, y = data

        check_data(x, y, splits)

        datasets = TSDatasets(x, y, splits=splits)

        return TSDataLoaders.from_dsets(
            datasets.train, datasets.valid, bs=[self._batch_size, self._batch_size],
            num_workers=0, pin_memory=True)

    def create_learner(self, loaders: TSDataLoaders, load: bool = False, predict: bool = False):
        learner = TSAILearner(dls=loaders, model=self._construct_model(loaders), early_stop_patience=self._early_stop_patience,
                              work_dir=self._dir, predict=predict)

        if load:
            learner.load(learner.save_model.fname, with_opt=True)

        return learner

    def decode_prediction(self, last_date, last_idx, inputs: Tensor, preds: Tensor, time_shift: int) -> DataFrame:

        inv_preds = self._dataset.inverse_transform(column=self._targets[0], x=preds)

        records = []

        for group_idx in range(inv_preds.shape[0]):
            group_input = inputs[group_idx].T
            group_preds = inv_preds[group_idx].T
            for time_idx in range(group_preds.shape[0]):
                current_date = last_date + timedelta(days=time_idx + time_shift)
                current_time_idx = last_idx + time_idx + time_shift
                time_preds = group_preds[time_idx]
                group_cat = group_input[time_idx][self._group_name_cat_idx].long().item()
                group_name = self._dataset.data_frame[self._group_name].cat.categories[group_cat]
                records.append(
                    {
                        'idx': current_time_idx,
                        'date': current_date,
                        self._group_name: group_name,
                        self._target_name_predict: np.round(time_preds).astype('int')
                    }
                )

        return DataFrame.from_records(records)

    def cutoff(self, window: int) -> DataFrame:
        cutoff = self._dataset.data_frame.idx.max() - window
        cutoff_data_frame = self._dataset.data_frame[lambda x: x.idx > cutoff]
        return cutoff_data_frame[['idx', 'date', self._group_name, self._target_name]]

    def run(self):
        logger.info(f'Running experiment: {self._name}')
        logger.info(f'Features: {",".join(self._features)}')
        logger.info(f'Targets: {",".join(self._targets)}')

        train, test, predict = self._dataset.get_splits(
            group_name=self._group_name,
            features=self._features,
            targets=self._targets,
            history_window=self._window,
            horizon=self._horizon)

        results: Dict[str, Any] = dict(
            lr=self._lr,
            model_class=self._model_class.__name__,
            name=self._name,
            window=self._window,
            horizon=self._horizon,
            features=",".join(self._features),
            targets=",".join(self._targets),
            batch_size=self._batch_size,
            early_stop_patience=self._early_stop_patience,
            max_epochs=self._epochs,
            country_filter=",".join(self._country_filter) if self._country_filter else 'all',
            region_filter=",".join(self._region_filter) if self._region_filter else 'all'
        )

        results.update(self.train(loaders=self.get_dls(data=train[0], splits=train[1])))

        test_results, test_inputs, test_predictions, test_targets = self.test(
            loaders=self.get_dls(data=test[0], splits=test[1])
        )

        results.update(test_results)

        results['is_early_stop'] = results['epochs'] < self._epochs

        test_dataframe = self.cutoff(self._horizon)
        test_dataframe_prediction = self.decode_prediction(
            last_date=test_dataframe.date.min(), last_idx=test_dataframe.idx.min(),
            inputs=test_inputs, preds=test_predictions, time_shift=0
        )

        results_dataframe = test_dataframe.merge(
            test_dataframe_prediction,
            on=['idx', 'date', self._group_name], how='left'
        )

        results_dataframe['absolute error'] = (
                results_dataframe[self._target_name_predict] - results_dataframe[self._target_name]
        ).abs()

        results_dataframe['absolute % error'] = (
                results_dataframe[self._target_name_predict] - results_dataframe[self._target_name]
        ).abs() / (results_dataframe[self._target_name].abs() + 1e-8) * 100

        results['Test MAE %'] = results_dataframe['absolute % error'].mean()
        results['Test MAE humans'] = int(results_dataframe['absolute error'].mean())

        forecast_inputs, forecast_preds, _ = self.predict(loaders=self.get_dls(data=predict[0], splits=predict[1]))

        forecast_dataframe = self.cutoff(self._window)

        forecast_dataframe_prediction = self.decode_prediction(
            last_date=forecast_dataframe.date.max(), last_idx=forecast_dataframe.idx.max(),
            inputs=forecast_inputs, preds=forecast_preds, time_shift=1
        )

        results_dataframe = results_dataframe.append(forecast_dataframe_prediction).set_index('idx')
        results_dataframe = results_dataframe.append(results_dataframe.describe(), ignore_index=False).fillna("")
        results_dataframe.index.name = 'idx'

        predict_path = os.path.join(self._dir, 'prediction.csv')
        results_dataframe.to_csv(predict_path)
        logger.info(f'Results:\n{json.dumps(results, indent=3, sort_keys=True)}')
        return results

    def train(self, loaders: TSDataLoaders):
        start_time = datetime.now()
        learner = self.create_learner(loaders=loaders)

        with learner.parallel_ctx():
            # learn.fine_tune(10)
            # print(r)
            # print(learn.loss_func)
            # logger.info('Finding LR')
            # lr_find = learner.lr_find()
            # logger.info(f'LR Find: {lr_find}')
            lr_find = None
            learner.fit_one_cycle(self._epochs, self._lr)
        duration = datetime.now() - start_time
        epochs = learner.recorder.log[0] + 1
        result = dict(
            lr_min=lr_find.lr_min if lr_find else None,
            lr_steep=lr_find.lr_steep if lr_find else None,
            epochs=epochs,
            duration=str(duration),
            epoch_time_secs=duration.total_seconds() / epochs
        )

        for idx, metric in enumerate(learner.recorder.metric_names[1:-1]):
            result[metric] = learner.recorder.save_model.final_record[idx]

        return result

    def test(self, loaders: TSDataLoaders):
        learner = self.create_learner(loaders=loaders, load=True)

        inputs, predictions, targets = learner.get_preds(with_input=True)

        result = dict()

        for idx, metric in enumerate(learner.recorder.metric_names[1:-1]):
            result[f'Test {metric}'] = learner.recorder.log[idx]

        return result, inputs, predictions, targets

    def predict(self, loaders: TSDataLoaders):
        learner = self.create_learner(loaders=loaders, load=True)
        return learner.get_preds(with_input=True)
