import json
from datetime import datetime
from logging import getLogger
from typing import List, Tuple, Optional, Dict, Any

import numpy as np
from tsai.data.core import TSDatasets, TSDataLoaders
from tsai.data.external import check_data

from covid19.datasets.rnbo import Rnbo
from .learner import TSAILearner

logger = getLogger(__name__)


class Experiment:
    def __init__(self, model_class: type, lr: float, early_stop_patience: int, epochs: int, features: List[str],
                 targets: List[str], window: int, horizon: int, batch_size: int, country_filter: List[str],
                 region_filter: Optional[List[str]] = None):
        self._lr: float = lr
        self._epochs: int = epochs
        self._name: str = datetime.now().strftime(f'%Y-%m-%d %H:%M {model_class.__name__}')
        self._features = features
        self._targets = targets
        self._window = window
        self._horizon = horizon
        self._batch_size = batch_size
        self._model_class = model_class
        self._early_stop_patience = early_stop_patience
        self._country_filter = country_filter
        self._region_filter = region_filter

    def _construct_model(self):
        return self._model_class(c_in=len(self._features), c_out=self._horizon)

    def get_dls(self, data: Tuple[np.ndarray, np.ndarray], splits: Tuple[np.ndarray, np.ndarray]) -> TSDataLoaders:
        x, y = data

        check_data(x, y, splits)

        datasets = TSDatasets(x, y, splits=splits)

        return TSDataLoaders.from_dsets(
            datasets.train, datasets.valid, bs=[self._batch_size, self._batch_size],
            num_workers=0, pin_memory=True)

    def run(self):
        logger.info(f'Running experiment: {self._name}')
        logger.info(f'Features: {",".join(self._features)}')
        logger.info(f'Targets: {",".join(self._targets)}')
        dataset = Rnbo()
        dataset.scrape(country_filter=self._country_filter, metrics=Rnbo.metrics, region_filter=self._region_filter)

        train, test = dataset.get_splits(
            group_name='country_region',
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

        results.update(
            self.train(loaders=self.get_dls(data=train[0], splits=train[1]))
        )

        results.update(
            self.test(loaders=self.get_dls(data=test[0], splits=test[1]))
        )

        results['is_early_stop'] = results['epochs'] < self._epochs
        logger.info(f'Results:\n{json.dumps(results, indent=3, sort_keys=True)}')
        return results

    def train(self, loaders: TSDataLoaders):
        start_time = datetime.now()
        learner = TSAILearner(dls=loaders, model=self._construct_model(), early_stop_patience=self._early_stop_patience,
                              experiment_name=self._name)

        with learner.parallel_ctx():
            # learn.fine_tune(10)
            # print(r)
            # print(learn.loss_func)
            lr_find = learner.lr_find()
            logger.info(lr_find)
            learner.fit_one_cycle(self._epochs, self._lr)
        duration = datetime.now() - start_time
        epochs = learner.recorder.log[0] + 1
        result = dict(
            lr_min=lr_find.lr_min,
            lr_steep=lr_find.lr_steep,
            epochs=epochs,
            duration=str(duration),
            epoch_time_secs=duration.total_seconds() / epochs
        )

        for idx, metric in enumerate(learner.recorder.metric_names[1:-1]):
            result[metric] = learner.recorder.save_model.final_record[idx]

        return result

    def test(self, loaders: TSDataLoaders):
        learner = TSAILearner(dls=loaders, model=self._construct_model(), early_stop_patience=self._early_stop_patience,
                              experiment_name=self._name)

        learner.load(learner.save_model.fname, with_opt=True)

        inputs, predictions, targets = learner.get_preds(with_input=True)

        result = dict()

        for idx, metric in enumerate(learner.recorder.metric_names[1:-1]):
            result[f'Test {metric}'] = learner.recorder.log[idx]

        return result

