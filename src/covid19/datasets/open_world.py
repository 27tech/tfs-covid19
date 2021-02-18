import requests
from logging import getLogger
from pathlib import Path
from covid19 import config
import os
import datetime
from pandas import DataFrame, read_csv

logger = getLogger(__name__)


class Dataset:
    _dataframe: DataFrame



class OpenWorldDataset(Dataset):
    _public_url: str = 'https://raw.githubusercontent.com/owid/covid-19-data/master/public/data/owid-covid-data.csv'
    _target_path: str

    def __init__(self):

        self._target_path = os.path.join(
            config.DATASETS_DIR, f'owid-covid-data_{datetime.datetime.now().date().isoformat()}.csv')

        if not os.path.exists(self._target_path):
            self.download()

        self._dataframe = read_csv(self._target_path)

    def download(self):
        logger.info(f'Downloading dataset: {self._public_url}')
        response = requests.get(self._public_url)
        response.raise_for_status()

        with open(self._target_path, 'wb') as output_file:
            for chunk in response.iter_content(chunk_size=4096):
                output_file.write(chunk)
