from typing import Optional

import requests
from logging import getLogger
from covid19 import config
import os
from datetime import date, datetime
from pandas import DataFrame, read_csv, PeriodIndex, read_pickle, to_datetime
import tempfile

logger = getLogger(__name__)


class Dataset:
    _dataframe: DataFrame

    @property
    def metrics(self):
        raise NotImplementedError()


class OpenWorldDataset(Dataset):
    _public_url: str = 'https://raw.githubusercontent.com/owid/covid-19-data/master/public/data/owid-covid-data.csv'

    def __init__(self, path: Optional[str] = None, start_date: Optional[datetime] = datetime(year=2020, month=3, day=1)):

        if path is None:
            path = os.path.join(config.DATASETS_DIR, f'owid-covid-data_{datetime.now().date().isoformat()}')

        path_pkl = f'{path}.pkl'
        path_csv = f'{path}.csv'

        if not os.path.exists(path_csv):
            self.download_to(path_csv)

        if not os.path.exists(path_pkl):
            dataframe = read_csv(path_csv)
            dataframe.date = to_datetime(dataframe.date)
            dataframe.date.index = PeriodIndex(dataframe.date, freq="D", name="Period")
            dataframe.to_pickle(path_pkl)
            self._dataframe = dataframe
        else:
            self._dataframe = read_pickle(path_pkl)

        if start_date:
            self._dataframe = self._dataframe[lambda x: x.date >= start_date]
        logger.info(f'Dataset range: [{self._dataframe.date.min().date()} {self._dataframe.date.max().date()}]')
        self._metrics = self._dataframe.columns.values[4:]

    def download_to(self, path: str):
        logger.info(f'Downloading dataset: {self._public_url}')
        response = requests.get(self._public_url)
        response.raise_for_status()

        with open(path, 'wb') as output_file:
            for chunk in response.iter_content(chunk_size=4096):
                output_file.write(chunk)

    @property
    def metrics(self):
        return self._metrics
