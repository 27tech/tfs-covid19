import calendar
from collections import deque
from pathlib import Path
from typing import List, Dict, Optional, Set

import requests
from logging import getLogger
from datetime import datetime, timedelta
import os
import json
from pprint import pprint
import pandas as pd

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
    _start_date = datetime(year=2020, month=3, day=1)
    _series = frozenset(
        [
            # 'confirmed',
            # 'deaths',
            # 'recovered',
            'existing',
            # 'suspicion',
            'delta_confirmed',
            # 'delta_deaths',
            # 'delta_recovered',
            # 'delta_existing',
            # 'delta_suspicion',
        ]
    )

    def __init__(self, path: str):
        self._path = path

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

    def prepare(self, metrics: Set[str], country_filter: Optional[List[str]] = None):

        columns = {
            'idx': deque(),
            'date': deque(),
            'country': deque(),
            'region': deque(),
            # 'series': deque(),
            # 'weekday': deque(),
            # 'value': deque()
        }
        for metric_name in metrics:
            assert metric_name in self._series
            columns[metric_name] = deque()

        # for weekday in list(calendar.day_name):
        #     columns[weekday] = deque()

        for idx, info in enumerate(self.download()):
            file_name, date = info
            with open(file_name, 'rb') as file:
                data: Dict[str] = json.load(file)
            # world_data = data['world']
            # world_data.append(
            #     {
            #         'country': 'Ukraine',
            #         'regions': data['ukraine']
            #     }
            # )
            transformed_data = deque([
                {
                    'country': 'Ukraine',
                    'regions': data['ukraine']
                }
            ])
            for country_data in data['world']:
                all_region_data = {'region': { 'label': {'en': 'all'}}}
                all_region_data.update(country_data)
                transformed_data.append(
                    {
                        'country': country_data['country'],
                        'regions': [all_region_data]
                    }
                )
            # del data
            for country_data in transformed_data:
                country_name = country_data['country']
                if country_filter and country_name not in country_filter:
                    continue

                for region_data in country_data['regions']:
                    columns['idx'].append(idx)
                    columns['date'].append(date)
                    columns['country'].append(country_name)
                    columns['region'].append(region_data['label']['en'])
                    for metric_name in metrics:
                        columns[metric_name].append(region_data[metric_name])
                    # columns['series'].append(sname)
                    # columns['value'].append(region_data[sname])
                        # current_weekday = date.strftime('%A')
                        # for weekday in list(calendar.day_name):
                        #     columns[weekday].append(current_weekday if current_weekday == weekday else "-")

        df = pd.DataFrame.from_dict(columns)
        df['date'] = pd.to_datetime(df['date'])
        logger.info(f"Dataset range: {df['date'].min()} - {df['date'].max()}")
        print(df.head(20))
        # for weekday in list(calendar.day_name):
        #     df[weekday] = df[weekday].astype("category")
            # df.set_index(weekday)
        # data[list(calendar.day_name)] = data[list(calendar.day_name)].astype("category")
        df.set_index('idx')
        print(df.head(20))
        return df
        columns: Dict[str, deque] = {
            'date': deque(),
            'idx': deque()
        }

        for region_id in regions_set:
            columns[region_id.__str__()] = deque()

        for idx, file_date in enumerate(sorted(files_data.keys()), 0):
            columns['date'].append(file_date)
            columns['idx'].append(idx)
            data = files_data[file_date]
            row_data = dict()
            for country_name in data.keys():
                if country_name not in country_filter:
                    continue

                regions_list: List[dict] = data[country_name]

                for region_data in regions_list:
                    row_data[str(region_data['id'])] = region_data['delta_confirmed']

                for region_id in regions_set:
                    if str(region_id) not in row_data:
                        row_data[str(region_id)] = 0

                for region_id in row_data:
                    columns[region_id].append(row_data[region_id])

        df = pd.DataFrame(columns)
        logger.info(f"Dataset range: {df['date'].min()} - {df['date'].max()}")
        df.set_index('idx')
        print(df)
        return df, [ str(s) for s in regions_set]
        # for country_id in regions_dict:
        #     for region_id = regions_dict[country_id]
        #
        #         #     row.update({
        #         #         'country_id': country_id,
        #         #         'region_id': region_id,
        #         #         'confirmed': region_data['confirmed'],
        #         #         'deaths': region_data['deaths'],
        #         #         'recovered': region_data['recovered'],
        #         #         'existing': region_data['existing'],
        #         #         'suspicion': region_data['suspicion'],
        #         #         'delta_confirmed': region_data['delta_confirmed'],
        #         #         'delta_deaths': region_data['delta_deaths'],
        #         #         'delta_recovered': region_data['delta_recovered'],
        #         #         'delta_existing': region_data['delta_existing'],
        #         #         'delta_suspicion': region_data['delta_suspicion']
        #         #     })
        #         #
        #         # pprint(row)
        #         # rows.append(row)
        #         # break
        # df = pd.DataFrame.from_dict(data=rows)

