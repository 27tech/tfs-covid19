from typing import List, Optional

from .experiment import Experiment
import pandas as pd
from logging import getLogger

import os

logger = getLogger(__name__)


class ExperimentSet:
    def __init__(self, models: List[type], lr: List[float], features: List[List[str]], targets: List[List[str]],
                 window: List[int], horizon: List[int], batch_size: List[int], country_filter: List[List[str]],
                 region_filter: List[Optional[List[str]]], early_stop_patience: int, epochs: int, runs: int,
                 do_predict: bool):
        self._models = models
        self._lr = lr
        self._features = features
        self._targets = targets
        self._window = window
        self._horizon = horizon
        self._batch_size = batch_size
        self._country_filter = country_filter
        self._region_filter = region_filter
        self._early_stop_patience = early_stop_patience
        self._epochs = epochs
        self._runs = runs
        self._do_predict = do_predict

    def run(self):
        fname = 'experiments.csv'
        # if os.path.exists(fname):
        #     df = pd.read_csv(fname)
        #     records = df.to_records()
        # else:
        records = []

        experiment_id = len(records) + 1
        for model in self._models:
            for lr in self._lr:
                for features in self._features:
                    for targets in self._targets:
                        for window in self._window:
                            for horizon in self._horizon:
                                for batch_size in self._batch_size:
                                    for country_filter in self._country_filter:
                                        for region_filter in self._region_filter:
                                            for r in range(self._runs):
                                                exp = Experiment(
                                                    model_class=model, lr=lr, features=features, targets=targets,
                                                    window=window, horizon=horizon, batch_size=batch_size,
                                                    country_filter=country_filter, region_filter=region_filter,
                                                    early_stop_patience=self._early_stop_patience,
                                                    epochs=self._epochs, do_predict=self._do_predict
                                                )
                                                e_results = exp.run()
                                                e_results['experiment_id'] = experiment_id
                                                records.append(e_results)
                                                all_results = pd.DataFrame.from_records(records).set_index('experiment_id')
                                                all_results.to_csv(fname)
                                                describe = all_results.describe()
                                                describe.to_csv('experiments_describe.csv')
                                                # log_df = all_results[['experiment_id', '','Final mape']]
                                                logger.info(f'Results:\n{all_results}')
                                                logger.info(f'Describe:\n{all_results.describe()}')
                                                experiment_id += 1
