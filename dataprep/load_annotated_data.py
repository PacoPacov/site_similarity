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


def load_splits(split_file):
    with open(os.path.join(_ANNOTATED_DATA_PATH, split_file)) as f:
        splits = json.load(f)

    return splits


def apply_splits(data, split_file):
    with open(os.path.join(_ANNOTATED_DATA_PATH, split_file)) as f:
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
