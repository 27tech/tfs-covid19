import json
import os
from collections import deque
from datetime import timedelta, datetime
from logging import getLogger
from typing import Dict, Set, Optional, List, Iterable, Any, Tuple

import pandas as pd
import requests
import calendar
from .dataset import Dataset

from .. import config

logger = getLogger(__name__)


class Rnbo(Dataset):
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

    def __init__(self, start_date=datetime(year=2020, month=3, day=1)):
        super().__init__()

        self._start_date = start_date

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


