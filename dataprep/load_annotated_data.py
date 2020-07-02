import csv
import os
import json

_ANNOTATED_DATA_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'annotated_data')


def load_corpus(corpus_file):
    corpus = []
    with open(os.path.join(_ANNOTATED_DATA_PATH, corpus_file)) as f:
        csv_reader = csv.DictReader(f)

        for row in csv_reader:
            corpus.append(row)

    return corpus


def apply_splits(data):
    with open(os.path.join(_ANNOTATED_DATA_PATH, 'splits.json')) as f:
        splits = json.load(f)

    data_splits = {}
    for index, split in enumerate(splits, start=0):
        test_urls = split[f'test-{index}'].split('\n')
        train_urls = split[f'train-{index}'].split('\n')

        data_splits[f'test-{index}'] = []
        for url in test_urls:
            for row in data:
                if url == row['source_url_processed']:
                    data_splits[f'test-{index}'].append(row)

        data_splits[f'train-{index}'] = []
        for url in train_urls:
            for row in data:
                if url == row['source_url_processed']:
                    data_splits[f'train-{index}'].append(row)

    return data_splits


if __name__ == '__main__':
    DATA = load_corpus()
    DATA_SPLITS = apply_splits(DATA)

    # save the splits
    for key in DATA_SPLITS:
        output_path = os.path.join(_ANNOTATED_DATA_PATH, 'splits', f'{key}.json')
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(DATA_SPLITS[key], f, indent=4)
