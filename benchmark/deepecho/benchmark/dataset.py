import os

import pandas as pd
from sdv import Metadata


from io import BytesIO
from urllib.parse import urljoin
from urllib.request import urlopen
from zipfile import ZipFile

import boto3
import botocore
import botocore.config

BUCKET_NAME = 'deepecho-data'
DATA_URL = 'http://{}.s3.amazonaws.com/'.format(BUCKET_NAME)
DATA_DIR = os.path.expanduser('~/deepecho_data')


def get_client():
    if boto3.Session().get_credentials():
        # credentials available and will be detected automatically
        config = None
    else:
        # no credentials available, make unsigned requests
        config = botocore.config.Config(signature_version=botocore.UNSIGNED)

    return boto3.client('s3', config=config)


class Dataset:
    """Dataset abstraction for benchmarking.

    This class loads as TimeSeries dataset from an sdv.Metadata
    in the format expected by DeepEcho models.

    It handles the extraction of the context columns from analyzing
    the data and identifying the columns that are constant for each
    entity_id.

    Args:
        dataset_path (str):
            Path to the dataset folder, where the metadata.json can be
            found.
        max_entities (int):
            Optionally restrict the number of entities to the indicated
            amount. If not given, use all the entities from the dataset.
    """
    def _load_table(self):
        tables = self.metadata.get_tables()
        if len(tables) > 1:
            raise ValueError('Only 1 table datasets are supported')

        self.table = tables[0]
        self.data_columns = self.metadata.get_fields(self.table)
        self.data = self.metadata.load_table(self.table)[self.data_columns]

    def _get_entity_columns(self):
        primary_key = self.metadata.get_primary_key(self.table)
        if not isinstance(primary_key, list):
            primary_key = [primary_key]

        return primary_key

    def _get_context_columns(self):
        context_columns = []
        for column in set(self.data.columns) - set(self.entity_columns):
            def is_constant(df):
                return len(df[column].unique()) == 1

            if self.data.groupby(self.entity_columns).apply(is_constant).all():
                context_columns.append(column)

        return context_columns

    def _ensure_downloaded(self):
        self.dataset_path = os.path.join(DATA_DIR, self.name)
        if not os.path.exists(self.dataset_path):
            client = get_client()
            os.makedirs(DATA_DIR, exist_ok=True)
            with urlopen(urljoin(DATA_URL, dataset_name + '.zip')) as fp:
                with ZipFile(BytesIO(fp.read())) as zipfile:
                    zipfile.extractall(self.dataset_path)

    def __init__(self, dataset, max_entities=None):
        if os.path.isdir(dataset):
            self.name = dataset
            self.dataset_path = dataset
        else:
            self.name = dataset
            self._ensure_downloaded()

        self.metadata = Metadata(os.path.join(self.dataset_path, 'metadata.json'))

        self._load_table()
        self.entity_columns = self._get_entity_columns()
        self.context_columns = self._get_context_columns()

        if max_entities:
            entities = self.data[self.entity_columns].drop_duplicates()
            if (max_entities < len(entities)):
                entities = entities.sample(max_entities)

                data = pd.DataFrame()
                for _, row in entities.iterrows():
                    mask = [True] * len(self.data)
                    for column in self.entity_columns:
                        mask &= self.data[column] == row[column]

                    data = data.append(self.data[mask])

                self.data = data

    def __repr__(self):
        return "Dataset('{}')".format(self.name)


def get_datasets_list():
    datasets = []
    client = boto3.client('s3')
    for dataset in client.list_objects(Bucket=BUCKET_NAME)['Contents']:
        key = dataset['Key']
        if key.endswith('.zip'):
            datasets.append(key.replace('.zip', ''))

    return datasets


def load_dataset(dataset_name):
    client = boto3.client('s3')
    dataset_path = os.path.join(DATA_DIR, dataset_name)
    if not os.path.exists(dataset_path):
        os.makedirs(DATA_DIR, exist_ok=True)
        with urlopen(urljoin(DATA_URL, dataset_name + '.zip')) as fp:
            with ZipFile(BytesIO(fp.read())) as zipfile:
                zipfile.extractall(dataset_path)

    return Dataset(dataset_path)
