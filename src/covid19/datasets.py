import calendar
from collections import deque
from pathlib import Path
from typing import List, Dict, Optional, Set, Any, Iterable, Deque

import requests
from logging import getLogger
from datetime import datetime, timedelta
import os
import json
from pprint import pprint
import pandas as pd
from . import config

logger = getLogger(__name__)


class OpenWorldDataset:
    _public_url: str = 'https://raw.githubusercontent.com/owid/covid-19-data/master/public/data/owid-covid-data.csv'

    def __init__(self):
        pass

    @classmethod
    def download(cls, target_path: Path):
        logger.info(f'Downloading dataset: {cls._public_url}')
        response = requests.get(cls._public_url)
        response.raise_for_status()
        with open(target_path, 'wb') as output_file:
            for chunk in response.iter_content(chunk_size=4096):
                output_file.write(chunk)


class RnboGovUa:
    _root_url = 'https://api-covid19.rnbo.gov.ua/data'
    _start_date = datetime(year=2020, month=4, day=1)
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
        ]
    )

    def __init__(self):
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

    def prepare_flat(self, country_filter: Optional[List[str]] = None):

        columns = {
            'idx': deque(),
            'date': deque(),
            'country': deque(),
            'region': deque(),
            'metric': deque(),
            # 'weekday': deque(),
            'value': deque()
        }

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
                    for metric_name in self.metrics:
                        columns['idx'].append(idx)
                        columns['date'].append(date)
                        columns['country'].append(country_name)
                        columns['region'].append(region_data['label']['en'])
                        columns['metric'].append(metric_name)
                        columns['value'].append(region_data[metric_name])

        df = pd.DataFrame.from_dict(columns)
        df.date = pd.to_datetime(df.date)
        df.date.index = pd.PeriodIndex(df.date, freq="D", name="Period")
        df.country = df.country.astype('category')
        df.region = df.region.astype('category')
        df.metric = df.metric.astype('category')
        logger.info(f"Dataset range: {df.date.min()} - {df.date.max()}")
        return df

    def data_frame_to_numpy(self, df: pd.DataFrame):
        shape = (
            df.date.unique().size,
            df.country.cat.categories.size,
            df.region.cat.categories.size,
            df.metric.cat.categories.size
        )
        import numpy as np
        array = np.zeros(shape)
        df['country'] = df.country.cat.codes
        df['region'] = df.region.cat.codes
        df['metric'] = df.metric.cat.codes
        for date_idx in sorted(df.idx.unique()):
            df_date = df[df.idx == date_idx]
            for idx, row in df_date.iterrows():
                array[date_idx, row.country, row.region, row.metric] = row.value
        return array

    def prepare_numpy(self, x_metrics: Set[str], y_metrics: Set[str], country_filter: Optional[List[str]] = None):
        df = self.prepare_flat(country_filter=country_filter)

        df_x = df.loc[df['metric'].isin(x_metrics)]
        x = self.data_frame_to_numpy(df_x)
        df_y = df.loc[df['metric'].isin(y_metrics)]
        y = self.data_frame_to_numpy(df_y)
        return x, y

    def prepare(self, metrics: Set[str], country_filter: Optional[List[str]] = None):

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
        # df.metric = df.metric.astype('category')
        df.country_region = df.country_region.astype('category')
        df['country_cat'] = df.country.cat.codes
        df['region_cat'] = df.region.cat.codes
        logger.info(f"Dataset range: {df['date'].min()} - {df['date'].max()}")
        # df = df.set_index('idx')
        return df
