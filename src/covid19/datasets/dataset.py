from datetime import timedelta
from typing import Tuple, List, Iterable, Optional
import numpy as np
from pandas import DataFrame
from logging import getLogger
from tsai.data.preparation import SlidingWindow
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, Normalizer, MaxAbsScaler, PowerTransformer

logger = getLogger(__name__)


class Dataset:
    _dataframe: DataFrame

    def __init__(self):
        self._scalers = dict()

    @property
    def metrics(self):
        raise NotImplementedError()

    @property
    def data_frame(self):
        return self._dataframe

    def normalize(self):

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
            # 'norm': Normalizer,
            'xabs': MaxAbsScaler,
            # 'pt': PowerTransformer

        }
        self._scalers = dict()
        for col in self.metrics:
            # self._dataframe[col] = self._dataframe[col]
            values = self._dataframe[col].values.reshape(-1, 1)
            for scaler_class in scalers_classes:
                scaler = scalers_classes[scaler_class]()
                scaled_values = scaler.fit_transform(values)
                scaled_column = f'{col}_{scaler_class}'
                self._dataframe[scaled_column] = scaled_values.reshape(-1)
                self._scalers[scaled_column] = scaler

        columns = self._dataframe[self.metrics].columns
        values = self._dataframe[self.metrics].values.T
        for scaler_class in scalers_classes:
            scaler = scalers_classes[scaler_class]()
            scaled_values = scaler.fit_transform(values)
            for k in range(scaled_values.shape[0]):
                scaled_column = f'{columns[k]}_{scaler_class}all'
                self._dataframe[scaled_column] = scaled_values[k]
                self._scalers[scaled_column] = scaler
        return self._scalers

    def inverse_transform(self, column: str, x: np.ndarray):
        inv = self._scalers[column].inverse_transform(x.reshape(-1, 1)).reshape(-1)
        assert inv.shape == x.shape
        return inv

    # noinspection PyMethodMayBeStatic
    def _sliding_window(self, group_name: str, features: List[str], targets: Optional[List[str]], dataframe: DataFrame,
                        history_window: int, horizon: int, stride: int, splits: int = 0, skip_missing=False) -> \
            Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]:

        sw = SlidingWindow(window_length=history_window, seq_first=True,
                           get_x=features, get_y=targets, stride=stride, horizon=horizon)

        time_steps = len(dataframe.date.unique())

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
            x = x.astype(np.float32)

            x_train.append(x[:-splits])
            x_valid.append(x[-splits:])

            if targets:
                y = y.astype(np.float32)
                y_train.append(y[:-splits])
                y_valid.append(y[-splits:])

        if targets:
            y_valid = np.vstack(y_valid)
            y_train = np.vstack(y_train)

        x_valid = np.vstack(x_valid)
        x_train = np.vstack(x_train)

        x_all = np.vstack([x_train, x_valid])

        y_all = None

        if targets:
            y_all = np.vstack([y_train, y_valid])
            assert len(y_valid) == splits * len(dataframe[group_name].unique())

        validation_steps = len(x_valid)
        total_indexes = list(range(x_all.shape[0]))
        split_indexes = np.asarray(total_indexes[:-validation_steps]), np.asarray(total_indexes[-validation_steps:])

        return (x_all, y_all), split_indexes

    def prepare(self, history_window: int, horizon: int) -> Tuple[DataFrame, DataFrame, DataFrame]:
        assert self._dataframe is not None, 'Scrape or load dataset before'

        # training_cutoff = self._dataframe.date.max() - timedelta(days=horizon)
        # train_dataframe = self._dataframe[lambda x: x.date < training_cutoff]
        train_dataframe = self._dataframe.copy()

        testing_cutoff = self._dataframe.date.max() - timedelta(days=history_window + horizon)
        test_dataframe = self._dataframe[lambda x: x.date >= testing_cutoff]

        predict_cutoff = self._dataframe.date.max() - timedelta(days=history_window)
        predict_dataframe = self._dataframe[lambda x: x.date >= predict_cutoff]

        return train_dataframe, test_dataframe, predict_dataframe

    def get_splits(self, group_name: str, features: List[str], targets: List[str], history_window: int, horizon: int):
        self.normalize()
        train, test, predict = self.prepare(history_window=history_window, horizon=horizon)
        print_columns = ['date'] + features + targets
        logger.info(f'Train Data Head:\n{train[print_columns].head(5)}')
        logger.info(f'Train Data Tail:\n{train[print_columns].tail(5)}')
        logger.info(f'Test Data Head:\n{test[print_columns].head(5)}')
        logger.info(f'Predict Data Head:\n{predict[print_columns].head(5)}')

        train_data, train_splits = self._sliding_window(
            group_name=group_name, features=features, targets=targets, history_window=history_window, horizon=horizon,
            stride=1, splits=horizon * 2, dataframe=train
        )
        train_splits_x = []
        for i in range(10):
            train_splits_x.extend(train_splits[0])
        train_splits = (train_splits_x, [train_splits[1][-1]])

        test_data, test_splits = self._sliding_window(
            group_name=group_name, features=features, targets=targets, history_window=history_window, horizon=horizon,
            stride=1, splits=1, dataframe=test
        )

        predict_data, predict_splits = self._sliding_window(
            group_name=group_name, features=features, targets=None, history_window=history_window, horizon=0,
            stride=1, splits=1, dataframe=predict
        )

        final_train_data, final_train_splits = self._sliding_window(
            group_name=group_name, features=features, targets=targets, history_window=history_window, horizon=horizon,
            stride=1, splits=1, dataframe=self._dataframe
        )

        return (train_data, train_splits), (test_data, test_splits), (predict_data, predict_splits), (final_train_data, final_train_splits)
