"""Dataset abstraction for benchmarking."""

import os
from io import BytesIO
from urllib.parse import urljoin
from urllib.request import urlopen
from zipfile import ZipFile

import boto3
import botocore
import botocore.config
import pandas as pd
from sdv import Metadata

BUCKET_NAME = 'deepecho-data'
DATA_URL = 'http://{}.s3.amazonaws.com/'.format(BUCKET_NAME)
DATA_DIR = os.path.expanduser('~/deepecho_data')


def _get_client():
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

    @staticmethod
    def _is_constant(column):
        def wrapped(group):
            return len(group[column].unique()) == 1

        return wrapped

    def _get_context_columns(self):
        context_columns = []
        for column in set(self.data.columns) - set(self.entity_columns):
            if self.data.groupby(self.entity_columns).apply(self._is_constant(column)).all():
                context_columns.append(column)

        return context_columns

    def _ensure_downloaded(self):
        self.dataset_path = os.path.join(DATA_DIR, self.name)
        if not os.path.exists(self.dataset_path):
            os.makedirs(DATA_DIR, exist_ok=True)
            with urlopen(urljoin(DATA_URL, self.name + '.zip')) as url:
                with ZipFile(BytesIO(url.read())) as zipfile:
                    zipfile.extractall(DATA_DIR)

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
            if max_entities < len(entities):
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


def _analyze_dataset(dataset_name):
    dataset = Dataset(dataset_name)
    groupby = dataset.data.groupby(dataset.entity_columns)
    sizes = groupby.size()
    return pd.Series({
        'entities': len(sizes),
        'entity_columns': len(dataset.entity_columns),
        'context_columns': len(dataset.context_columns),
        'data_columns': len(dataset.data_columns),
        'max_sequence_len': sizes.max(),
        'min_sequence_len': sizes.min(),
    })


def get_datasets_list(extended=False):
    """Get a list with the names of all the availale datasets."""
    datasets = []
    client = _get_client()
    for dataset in client.list_objects(Bucket=BUCKET_NAME)['Contents']:
        key = dataset['Key']
        if key.endswith('.zip'):
            datasets.append({
                'dataset': key.replace('.zip', ''),
                'size': dataset['Size']
            })

    datasets = pd.DataFrame(datasets).sort_values('size')
    if extended:
        details = datasets.dataset.apply(_analyze_dataset)
        datasets = pd.concat([datasets, details], axis=1)

    return datasets
