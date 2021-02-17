import json
import os
from collections import deque
from datetime import timedelta, datetime
from logging import getLogger
from typing import Dict, Set, Optional, List, Iterable, Any, Tuple

import pandas as pd
import requests
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, Normalizer, scale
from sklearn import preprocessing
from tsai.data.preparation import SlidingWindow
import numpy as np
import calendar

from .. import config

logger = getLogger(__name__)


class Rnbo:
    _root_url = 'https://api-covid19.rnbo.gov.ua/data'
    metrics = frozenset(
        [
            'confirmed',
            'deaths',
            'recovered',
            'existing',
            'suspicion',
            'delta_confirmed',
            'delta_deaths',
            'delta_recovered',
            'delta_existing',
            'delta_suspicion',
            'lat',
            'lng',
        ]
    )

    _dataframe: Optional[pd.DataFrame] = None
    _scalers: Dict[str, Any] = None

    def __init__(self, start_date=datetime(year=2020, month=3, day=1)):
        self._start_date = start_date
        self._scalers = dict()
        self._path = os.path.join(config.DATASETS_DIR, 'rnbo.gov.ua')

        if not os.path.exists(self._path):
            os.makedirs(self._path, exist_ok=True)

    def _download_date(self, date: datetime, output_path: str):
        logger.info(f"Download date: {date.strftime('%Y-%m-%d')}")
        response = requests.get(
            url=self._root_url,
            params={'to': date.strftime('%Y-%m-%d')}
        )
        response.raise_for_status()

        with open(output_path, 'wb') as file:
            for chunk in response.iter_content(chunk_size=4096):
                file.write(chunk)

    def _date_generator(self):
        utc_now = datetime.utcnow().date()
        current_date = self._start_date
        while current_date.date() < utc_now:
            yield current_date.date()
            current_date += timedelta(days=1)

    def download(self):
        current_date = self._start_date
        for date in self._date_generator():
            file_name = os.path.join(
                self._path,
                current_date.strftime('%Y-%m-%d') + '.json'
            )

            if not os.path.exists(file_name):
                self._download_date(current_date, output_path=file_name)
            yield file_name, date
            current_date += timedelta(days=1)

    def scrape(self, metrics: Set[str], country_filter: Optional[List[str]] = None,
               region_filter: Optional[List[str]] = None):

        columns = {
            'idx': deque(),
            'date': deque(),
            'country': deque(),
            'country_region': deque(),
            'region': deque(),
            # 'series': deque(),
            # 'weekday': deque(),
            # 'value': deque()
        }

        for metric_name in metrics:
            assert metric_name in self.metrics
            columns[metric_name] = deque()

        for idx, info in enumerate(self.download()):
            file_name, date = info
            with open(file_name, 'rb') as file:
                data: Dict[str] = json.load(file)
            transformed_data = deque([
                {
                    'country': 'Ukraine',
                    'regions': data['ukraine']
                }
            ])

            for country_data in data['world']:
                if country_data['country'] == 'Ukraine':
                    continue
                all_region_data = {'region': {'label': {'en': 'all'}}}
                all_region_data.update(country_data)
                transformed_data.append(
                    {
                        'country': country_data['country'],
                        'regions': [all_region_data]
                    }
                )
            # del datasets
            for country_data in transformed_data:
                country_name = country_data['country']
                if country_filter and country_name not in country_filter:
                    continue

                for region_data in country_data['regions']:
                    columns['idx'].append(idx)
                    columns['date'].append(date)
                    columns['country'].append(country_name)
                    columns['region'].append(region_data['label']['en'])
                    columns['country_region'].append(f"{country_name}_{region_data['label']['en']}")
                    for metric_name in metrics:
                        columns[metric_name].append(float(region_data[metric_name]))

        df = pd.DataFrame.from_dict(columns)
        df.date = pd.to_datetime(df.date)
        df.date.index = pd.PeriodIndex(df.date, freq="D", name="Period")
        df.country = df.country.astype('category')
        df.region = df.region.astype('category')
        df.country_region = df.country_region.astype('category')
        df['country_cat'] = df.country.cat.codes
        df['region_cat'] = df.region.cat.codes
        df['country_region_cat'] = df.country_region.cat.codes

        for idx, day_name in enumerate(calendar.day_name):
            df[day_name] = df['date'].apply(
                lambda x: 1. if x.day_name() == day_name else .0)

        logger.info(f"Dataset range: {df['date'].min()} - {df['date'].max()}")
        self._dataframe = df
        if region_filter is not None:
            self._dataframe = self._dataframe[self._dataframe.region.isin(region_filter)]
        return self._dataframe

    @property
    def data_frame(self):
        return self._dataframe

    def normalize(self, metrics: Iterable[str]):

        class FakeScaler:
            def fit_transform(self, x):
                return x

            def inverse_transform(self, x):
                return np.asarray(x)

        scalers_classes = {
            'origin': FakeScaler,
            'nx': MinMaxScaler,
            'std': StandardScaler,
            'rob': RobustScaler,
            'norm': Normalizer,
        }
        self._scalers = dict()
        for col in metrics:
            values = self._dataframe[col].values.reshape(-1, 1).astype('float32')
            for scaler_class in scalers_classes:
                scaler = scalers_classes[scaler_class]()
                scaled_values = scaler.fit_transform(values)
                scaled_column = f'{col}_{scaler_class}'
                self._dataframe[scaled_column] = scaled_values.reshape(-1)
                self._scalers[scaled_column] = scaler

        columns = self._dataframe[metrics].columns
        values = self._dataframe[metrics].values.T.astype('float32')
        for scaler_class in scalers_classes:
            scaler = scalers_classes[scaler_class]()
            scaled_values = scaler.fit_transform(values)
            for k in range(scaled_values.shape[0]):
                scaled_column = f'{columns[k]}_{scaler_class}_all'
                self._dataframe[scaled_column] = scaled_values[k]
                self._scalers[scaled_column] = scaler
        return self._scalers

    def _sliding_window(self, group_name: str, features: List[str], targets: List[str], dataframe: pd.DataFrame,
                        history_window: int, horizon: int, stride: int, splits: int = 0, skip_missing=False) -> \
            Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]:

        sw = SlidingWindow(window_length=history_window, seq_first=True,
                           get_x=features, get_y=targets, stride=stride, horizon=horizon)

        time_steps = len(dataframe.idx.unique())

        x_train = []
        y_train = []
        x_valid = []
        y_valid = []

        for group in dataframe[group_name].unique():
            group_data = dataframe.loc[dataframe[group_name] == group]

            if len(group_data) != time_steps:
                if skip_missing:
                    logger.info(f'Skip: {group}')
                    continue
                assert False

            x, y = sw(group_data)
            y = y.astype('float32')
            x = x.astype('float32')

            x_train.append(x[:-splits])
            y_train.append(y.astype('float32')[:-splits])

            x_valid.append(x[-splits:])
            y_valid.append(y[-splits:])

        y_valid = np.vstack(y_valid)
        x_valid = np.vstack(x_valid)
        y_train = np.vstack(y_train)
        x_train = np.vstack(x_train)

        x_all = np.vstack([x_train, x_valid])
        y_all = np.vstack([y_train, y_valid])
        assert len(y_valid) == splits * len(dataframe[group_name].unique())
        validation_steps = len(y_valid)
        total_indexes = list(range(y_all.shape[0]))
        split_indexes = np.asarray(total_indexes[:-validation_steps]), np.asarray(total_indexes[-validation_steps:])

        return (x_all, y_all), split_indexes

    def prepare(self, history_window: int, horizon: int) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        assert self._dataframe is not None, 'Scrape or load dataset before'

        self.normalize(self.metrics)

        training_cutoff = self._dataframe.idx.max()
        train_dataframe = self._dataframe[self._dataframe.idx <= training_cutoff]

        testing_cutoff = self._dataframe.idx.max() - history_window - horizon
        test_dataframe = self._dataframe[lambda x: x.idx >= testing_cutoff]

        predict_cutoff = self._dataframe.idx.max() - history_window
        predict_dataframe = self._dataframe[lambda x: x.idx >= predict_cutoff]

        return train_dataframe, test_dataframe, predict_dataframe

    def get_splits(self, group_name: str, features: List[str], targets: List[str], history_window: int, horizon: int):
        train, test, predict = self.prepare(history_window=history_window, horizon=horizon)
        logger.info(f'Train Data Tail:\n{train.tail(5)}')
        logger.info(f'Test Data:\n{test}')
        logger.info(f'Predict Data:\n{predict}')

        train_data, train_splits = self._sliding_window(
            group_name=group_name, features=features, targets=targets, history_window=history_window, horizon=horizon,
            stride=1, splits=horizon, dataframe=train
        )

        test_data, test_splits = self._sliding_window(
            group_name=group_name, features=features, targets=targets, history_window=history_window, horizon=horizon,
            stride=1, splits=1, dataframe=test
        )

        predict_data, predict_splits = self._sliding_window(
            group_name=group_name, features=features, targets=targets, history_window=history_window, horizon=0,
            stride=1, splits=1, dataframe=predict
        )
        # Drop targets
        predict_data = (predict_data[0], None)

        return (train_data, train_splits), (test_data, test_splits), (predict_data, predict_splits)

    def inverse_transform(self, column: str, x: np.ndarray):
        return self._scalers[column].inverse_transform(x)
