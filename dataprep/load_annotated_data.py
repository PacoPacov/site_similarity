import csv
import os
import json

_ANNOTATED_DATA_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data', 'annotated_data')
_INVALID_URLS_2018 = os.path.join(_ANNOTATED_DATA_PATH, 'invalid_urls_2018.json')


def _load_invalid_urls_2018():
    with open(_INVALID_URLS_2018, 'r') as f:
        invalid_urls_2018 = json.load(f)

    return invalid_urls_2018


def load_corpus(corpus_file, data_year='2020', delimiter=','):
    corpus = []
    with open(os.path.join(_ANNOTATED_DATA_PATH, corpus_file)) as f:
        csv_reader = csv.DictReader(f, delimiter=delimiter)

        for row in csv_reader:
            corpus.append(row)

    if data_year == '2018':
        invalid_urls_2018 = _load_invalid_urls_2018()

        modified_corpus = []
        for row in corpus:
            if invalid_urls_2018.get(row['source_url_processed']):
                row['source_url_processed'] = invalid_urls_2018[row['source_url_processed']]

            modified_corpus.append(row)

        corpus = modified_corpus

    return corpus


def load_splits(split_file, data_year='2020'):
    with open(os.path.join(_ANNOTATED_DATA_PATH, split_file)) as f:
        splits = json.load(f)

    if data_year == '2018':
        invalid_urls_2018 = _load_invalid_urls_2018()

        modified_splits = []
        for index in range(len(splits)):

            train_index_urls = [invalid_urls_2018[train_url] if train_url in invalid_urls_2018 else train_url
                                for train_url in splits[index][f"train-{index}"].split('\n')]

            test_index_urls = [invalid_urls_2018[test_url] if test_url in invalid_urls_2018 else test_url
                               for test_url in splits[index][f"test-{index}"].split('\n')]

            modified_splits.append({f'test-{index}': '\n'.join(test_index_urls),
                                    f'train-{index}': '\n'.join(train_index_urls)})

        splits = modified_splits

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
