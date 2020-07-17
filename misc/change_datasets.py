import os
import json

datasets_path = 'datasets'
for dataset in os.listdir(datasets_path):
    path = os.path.join(datasets_path, dataset)
    metadata_path = os.path.join(path, 'metadata.json')
    with open(metadata_path) as f:
        metadata = json.load(f)

    table = metadata['tables'].pop('data')
    fields = table['fields']
    del fields['s_index']
    del fields['tt_split']

    table['path'] = dataset + '.csv'
    metadata['tables'][dataset] = table

    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=4)
