"""
This script converts the data from the TSC repository into our user data
format. It was used to generate the datasets on S3 that are now being used
for benchmarking.

1. Download the TSFile datasets from the TSC repository.

    http://www.timeseriesclassification.com/Downloads/Archives/Multivariate2018_ts.zip

2. Process them with this script.

3. Upload the data to S3

    aws s3 sync ~/Desktop/datasets s3://deepecho-data/ --exclude="*" --include="*.zip"
"""
import os
import json
import pandas as pd
from glob import glob
from tqdm import tqdm
from sdv import Metadata
import shutil
from sktime.utils.load_data import load_from_tsfile_to_dataframe

def to_our_format(path_to_train, path_to_test):
    dfs = []

    train_x, train_y = load_from_tsfile_to_dataframe(path_to_train)
    for (idx, x), y in zip(train_x.iterrows(), train_y):
        df = pd.DataFrame()
        for col_name, col_value in x.iteritems():
            df[col_name] = col_value
        df.insert(0, "tt_split", [1]*len(df))
        df.insert(0, "s_index", range(len(df)))
        df.insert(0, "e_id", idx)
        df.insert(0, "ml_class", y)
        dfs.append(df)
    idx_offset = idx + 1

    test_x, test_y = load_from_tsfile_to_dataframe(path_to_test)
    for (idx, x), y in zip(test_x.iterrows(), test_y):
        df = pd.DataFrame()
        for col_name, col_value in x.iteritems():
            df[col_name] = col_value
        df.insert(0, "tt_split", [0]*len(df))
        df.insert(0, "s_index", range(len(df)))
        df.insert(0, "e_id", idx_offset + idx)
        df.insert(0, "ml_class", y)
        dfs.append(df)

    return pd.concat(dfs)


for path_to_train in tqdm(glob("Multivariate_ts/**/*_TRAIN.ts")):
    dataset_name = os.path.basename(path_to_train)
    dataset_name = dataset_name.replace(".ts", "")
    dataset_name = dataset_name.replace("_TRAIN", "")
    dataset_dir = "datasets/%s" % dataset_name
    os.makedirs(dataset_dir, exist_ok=True)

    path_to_test = path_to_train.replace("_TRAIN", "_TEST")
    path_to_csv = os.path.join(dataset_dir, "%s.csv" % dataset_name)
    path_to_metadata = os.path.join(dataset_dir, "metadata.json")
    path_to_readme = os.path.join(dataset_dir, "README.md")
    print(path_to_csv, path_to_metadata, path_to_readme)

    df = to_our_format(path_to_train, path_to_test)
    df.to_csv(path_to_csv, index=False)

    metadata = Metadata()
    metadata.add_table('data', data=df, primary_key='e_id')
    metadata.to_json(path_to_metadata)

    with open(os.path.join(dataset_dir, "task.json"), "wt") as fout:
        json.dump({
            "task_type": "classification",
            "key": ["e_id"],
            "target": "ml_class",
            "ignored": ["tt_split", "s_index"]
        }, fout)

    with open(path_to_readme, "wt") as fout:
        fout.write("""# %s

This dataset originates from the Time Series Classification
dataset repository (http://www.timeseriesclassification.com/).

It was converted into this format on June 28th, 2020.""" % (dataset_name))


    shutil.make_archive(dataset_dir, 'zip', dataset_dir)
