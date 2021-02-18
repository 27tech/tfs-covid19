from typing import Optional, List, Tuple

import requests
from logging import getLogger
from covid19 import config
import os
from datetime import date, datetime
from pandas import DataFrame, read_csv, PeriodIndex, read_pickle, to_datetime
import numpy as np
from .dataset import Dataset

logger = getLogger(__name__)


class OpenWorldDataset(Dataset):
    _public_url: str = 'https://raw.githubusercontent.com/owid/covid-19-data/master/public/data/owid-covid-data.csv'

    def __init__(self, path: Optional[str] = None, start_date: Optional[datetime] = datetime(year=2020, month=3, day=1)):
        super().__init__()

        if path is None:
            path = os.path.join(config.DATASETS_DIR, f'owid-covid-data_{datetime.now().date().isoformat()}')

        path_pkl = f'{path}.pkl'
        path_csv = f'{path}.csv'

        if not os.path.exists(path_csv):
            self.download_to(path_csv)

        if not os.path.exists(path_pkl):
            dataframe: DataFrame = read_csv(path_csv)
            dataframe.date = to_datetime(dataframe.date)
            dataframe = dataframe.fillna(.0).drop(columns=['tests_units'])
            dataframe['location'] = dataframe['location'].astype('category')
            dataframe.sort_values(by=['date', 'location'], inplace=True)
            dataframe.to_pickle(path_pkl)

        self._dataframe = read_pickle(path_pkl)

        if start_date:
            self._dataframe = self._dataframe[lambda x: x.date >= start_date]

        self._dataframe.date.index = PeriodIndex(self._dataframe.date, freq="D", name="Period")
        self.update_locations()

        self._dataframe['total_cases_per_population'] = self._dataframe['total_cases'] / self._dataframe['population']
        self._dataframe['new_cases_per_population'] = self._dataframe['new_cases'] / self._dataframe['population']

        logger.info(f'Dataset range: [{self._dataframe.date.min().date()} {self._dataframe.date.max().date()}]')
        self._metrics = self._dataframe.columns.values[4:]

    def update_locations(self):
        self._dataframe['location'].cat.remove_unused_categories(inplace=True)
        self._dataframe['location_cat'] = self._dataframe['location'].values.codes

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

    def filter_country(self, countries: List[str]):
        self._dataframe = self._dataframe[self._dataframe['location'].isin(countries)]
        self.update_locations()

