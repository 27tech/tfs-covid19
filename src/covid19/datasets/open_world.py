import requests
from logging import getLogger
from pathlib import Path

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
