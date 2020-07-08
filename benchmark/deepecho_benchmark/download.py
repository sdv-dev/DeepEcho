import os
from io import BytesIO
from urllib.parse import urljoin
from urllib.request import urlopen
from zipfile import ZipFile

import boto3

BUCKET_NAME = 'deepecho-data'
DATA_URL = 'http://{}.s3.amazonaws.com/'.format(BUCKET_NAME)


def download(data_dir):
    """Download benchmark datasets from S3.

    This downloads the benchmark datasets from S3 into the target folder in an
    uncompressed format. It skips datasets that have already been downloaded.

    Args:
        data_dir: The directory to download the datasets to.

    Returns:
        A DataFrame describing the downloaded datasets.
    """
    datasets = []
    client = boto3.client('s3')
    for dataset in client.list_objects(Bucket=BUCKET_NAME)['Contents']:
        datasets.append(dataset)
        dataset_name = dataset['Key'].replace(".zip", "")
        dataset_path = os.path.join(data_dir, dataset_name)
        if os.path.exists(dataset_path):
            dataset["Status"] = "Skipped"
            print("Skipping %s" % dataset_name)
        else:
            dataset["Status"] = "Downloaded"
            print("Downloading %s" % dataset_name)
            with urlopen(urljoin(DATA_URL, dataset['Key'])) as fp:
                with ZipFile(BytesIO(fp.read())) as zipfile:
                    zipfile.extractall(dataset_path)
    return datasets
