"""Dataset abstraction for benchmarking."""

import os
import shutil
from io import BytesIO
from urllib.parse import urljoin
from urllib.request import urlopen
from zipfile import ZipFile

import boto3
import botocore
import botocore.config
import pandas as pd
from sdv import Metadata

from deepecho.sequences import assemble_sequences

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
        segment_size (int, pd.Timedelta or str):
            If specified, cut each training sequence in several segments of the
            indicated size. The size can either can passed as an integer value,
            which will interpreted as the number of data points to put on each
            segment, or as a pd.Timedelta (or equivalent str representation),
            which will be interpreted as the segment length in time. Timedelta
            segment sizes can only be used with sequence indexes of type datetime.
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
        candidate_columns = set(self.data.columns) - set(self.entity_columns)
        if self.entity_columns:
            for column in candidate_columns:
                if self.data.groupby(self.entity_columns).apply(self._is_constant(column)).all():
                    context_columns.append(column)

        else:
            for column in candidate_columns:
                if self._is_constant(self.data[column]):
                    context_columns.append(column)

        return context_columns

    def _ensure_downloaded(self):
        self.dataset_path = os.path.join(DATA_DIR, self.name)
        if not os.path.exists(self.dataset_path):
            os.makedirs(DATA_DIR, exist_ok=True)
            with urlopen(urljoin(DATA_URL, self.name + '.zip')) as url:
                with ZipFile(BytesIO(url.read())) as zipfile:
                    zipfile.extractall(DATA_DIR)

    def _filter_entities(self, max_entities):
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

    def _get_evaluation_data(self, segment_size):
        data_columns = [
            column for column in self.data
            if column not in self.entity_columns + self.context_columns
        ]
        sequences = assemble_sequences(
            self.data,
            self.entity_columns,
            self.context_columns,
            segment_size,
            self.sequence_index
        )
        evaluation_data = pd.DataFrame(columns=self.data.columns)
        for idx, sequence in enumerate(sequences):
            sequence_df = pd.DataFrame(sequence['data'], index=data_columns).T
            for column, value in zip(self.context_columns, sequence['context']):
                sequence_df[column] = value

            for column in self.entity_columns:
                sequence_df[column] = idx

            evaluation_data = evaluation_data.append(sequence_df)

        return evaluation_data

    def __init__(self, dataset, max_entities=None, segment_size=None):
        if os.path.isdir(dataset):
            self.name = dataset
            self.dataset_path = dataset
        else:
            self.name = dataset
            self._ensure_downloaded()

        self.metadata = Metadata(os.path.join(self.dataset_path, 'metadata.json'))

        self._load_table()

        properties = self.metadata._metadata.get('properties')   # pylint: disable=W0212
        if properties:
            self.entity_columns = properties['entity_columns']
            self.sequence_index = properties.get('sequence_index')
        else:
            self.entity_columns = self._get_entity_columns()
            self.sequence_index = None

        self.context_columns = self._get_context_columns()

        if max_entities:
            self._filter_entities(max_entities)

        if not segment_size:
            self.evaluation_data = self.data
        else:
            self.evaluation_data = self._get_evaluation_data(segment_size)

    def describe(self):
        """Describe this datasets.

        The output is a ``pandas.Series`` containing:
            * ``entities``: Number of entities in the dataset.
            * ``entity_colums``: Number of entity columns.
            * ``context_colums``: Number of context columns.
            * ``data_columns``: Number of data columns.
            * ``max_sequence_len``: Maximum sequence length.
            * ``min_sequence_len``: Minimum sequence length.

        Returns:
            pandas.Series
        """
        groupby = self.data.groupby(self.entity_columns)
        sizes = groupby.size()
        return pd.Series({
            'entities': len(sizes),
            'entity_columns': len(self.entity_columns),
            'context_columns': len(self.context_columns),
            'data_columns': len(self.data_columns),
            'max_sequence_len': sizes.max(),
            'min_sequence_len': sizes.min(),
        })

    def __repr__(self):
        return "Dataset('{}')".format(self.name)


def _describe_dataset(dataset_name):
    return Dataset(dataset_name).describe()


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
        details = datasets.dataset.apply(_describe_dataset)
        datasets = pd.concat([datasets, details], axis=1)

    return datasets


def make_dataset(name, data, datasets_path='.', entity_columns=None, sequence_index=None):
    """Make a Dataset from a DataFrame.

    Args:
        name (str):
            Name of this dataset.
        data (pandas.DataFrame or str):
            Data passed as a DataFrame or as a path to a CSV file.
        datasets_path (str):
            (Optional) Path to the folder in which a new folder will be created
            for this dataset. Defaults to the current working directory.
        entity_columns (list or None):
            (Optional) List of names of the columns that form the entity_id of this
            dataset. If ``None`` (default), no entity columns are set.
        sequence_index (str or None):
            (Optional) Name of the column that is the sequence index of this dataset.
    """
    if isinstance(data, str):
        data = pd.read_csv(data)

    base_path = os.path.join(datasets_path, name)
    if os.path.exists(base_path):
        shutil.rmtree(base_path)

    os.makedirs(base_path, exist_ok=True)

    cwd = os.getcwd()
    try:
        os.chdir(base_path)
        csv_name = name + '.csv'
        data.to_csv(csv_name, index=False)

        metadata = Metadata()
        metadata.add_table(name, csv_name)
        metadata._metadata['properties'] = {
            'entity_columns': entity_columns or [],
            'sequence_index': sequence_index,
        }
        metadata.to_json('metadata.json')
    finally:
        os.chdir(cwd)
