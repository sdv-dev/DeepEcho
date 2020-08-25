"""Dataset abstraction for benchmarking."""

import json
import logging
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

LOGGER = logging.getLogger(__name__)

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

    VERSION = '0.1.1'

    def _load_table(self):
        columns = list(self.metadata.get_fields(self.table_name).keys())
        primary_key = self.metadata.get_primary_key(self.table_name)
        if primary_key:
            columns.remove(primary_key)

        self.data = self.metadata.load_table(self.table_name)[columns]

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

    def _download(self):
        os.makedirs(DATA_DIR, exist_ok=True)
        filename = '{}_v{}.zip'.format(self.name, self.VERSION)
        url = urljoin(DATA_URL, filename)
        LOGGER.info('Downloading dataset %s from %s', self.name, url)
        with urlopen(url) as remote:
            with ZipFile(BytesIO(remote.read())) as zipfile:
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
        sequences = assemble_sequences(
            self.data,
            self.entity_columns,
            self.context_columns,
            segment_size,
            self.sequence_index
        )
        evaluation_data = pd.DataFrame(columns=self.data.columns)
        for idx, sequence in enumerate(sequences):
            sequence_df = pd.DataFrame(sequence['data'], index=self.model_columns).T
            for column, value in zip(self.context_columns, sequence['context']):
                sequence_df[column] = value

            for column in self.entity_columns:
                sequence_df[column] = idx

            evaluation_data = evaluation_data.append(sequence_df)

        return evaluation_data

    def _load_metadata(self):
        dataset_path = os.path.join(DATA_DIR, self.name)
        metadata_path = os.path.join(dataset_path, 'metadata.json')

        try:
            self.metadata = Metadata(metadata_path)
            version = self.metadata.get_table_meta(self.table_name)['deepecho_version']
            assert version == self.VERSION
        except Exception:
            self._download()
            self.metadata = Metadata(metadata_path)

    def __init__(self, dataset, table_name=None, max_entities=None, segment_size=None):
        if os.path.isdir(dataset):
            self.name = os.path.basename(dataset)
            self.table_name = table_name or self.name
            self.metadata = Metadata(os.path.join(dataset, 'metadata.json'))
        else:
            self.name = dataset
            self.table_name = table_name or self.name
            self._load_metadata()

        self._load_table()

        table_meta = self.metadata.get_table_meta(self.table_name)
        self.entity_columns = table_meta.get('entity_columns') or []
        self.sequence_index = table_meta.get('sequence_index')
        self.context_columns = self._get_context_columns()
        self.model_columns = [
            column for column in self.data.columns
            if column not in self.entity_columns + self.context_columns + [self.sequence_index]
        ]

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
            'model_columns': len(self.model_columns),
            'max_sequence_len': sizes.max(),
            'min_sequence_len': sizes.min(),
        })

    def __repr__(self):
        return "Dataset('{}')".format(self.name)


def get_datasets_list():
    """Get a list with the names and details of all the availale datasets."""
    url = urljoin(DATA_URL, 'datasets.csv')
    return pd.read_csv(url).sort_values('size_in_kb')


def make_dataset(name, data, table_name=None, entity_columns=None,
                 sequence_index=None, datasets_path='.', zipped=False):
    """Make a Dataset from a DataFrame.

    Args:
        name (str):
            Name of this dataset.
        data (pandas.DataFrame or str):
            Data passed as a DataFrame or as a path to a CSV file.
        table_name (str or None):
            Optionally give the table a different name.
        entity_columns (list or None):
            (Optional) List of names of the columns that form the entity_id of this
            dataset. If ``None`` (default), no entity columns are set.
        sequence_index (str or None):
            (Optional) Name of the column that is the sequence index of this dataset.
        datasets_path (str):
            (Optional) Path to the folder in which a new folder will be created
            for this dataset. Defaults to the current working directory.
        zipped (boolean):
            If true, compress the dataset folder using zip and remove the dataset folder.
    """
    if isinstance(data, str):
        data = pd.read_csv(data)

    base_path = os.path.join(datasets_path, name)
    if os.path.exists(base_path):
        shutil.rmtree(base_path)

    os.makedirs(base_path, exist_ok=True)

    table_name = table_name or name

    cwd = os.getcwd()
    try:
        os.chdir(base_path)
        csv_name = table_name + '.csv'
        data.to_csv(csv_name, index=False)

        metadata = Metadata()
        metadata.add_table(name, csv_name)
        meta_dict = metadata.to_dict()
        table_meta = meta_dict['tables'][table_name]
        table_meta['entity_columns'] = entity_columns or []
        table_meta['sequence_index'] = sequence_index
        table_meta['deepecho_version'] = Dataset.VERSION

        with open('metadata.json', 'w') as metadata_file:
            json.dump(meta_dict, metadata_file, indent=4)

        if zipped is False:
            LOGGER.info('Dataset %s generated in folder %s', name, base_path)
        else:
            os.chdir('..')
            shutil.make_archive(name, 'zip', name)
            shutil.rmtree(name)
            LOGGER.info('Zip file %s generated in folder %s', name, base_path)
    finally:
        os.chdir(cwd)
