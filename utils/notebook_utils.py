import csv
import json
import os

from bs4 import BeautifulSoup
import pandas as pd
import numpy as np
from stellargraph import StellarGraph, data
import networkx as nx
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score

from dataprep.load_annotated_data import _ANNOTATED_DATA_PATH

from dataprep.load_annotated_data import apply_splits, load_corpus, load_splits
_DATA_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')
_INVALID_URLS_2018 = os.path.join(_ANNOTATED_DATA_PATH, 'invalid_urls_2018.json')
_ALL_DATA = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'dataset', 'all_data')


def calculate_metrics(actual, predicted):
    """
    Calculate performance metrics given the actual and predicted labels.
    Returns the macro-F1 score, the accuracy, the flip error rate and the
    mean absolute error (MAE).
    The flip error rate is the percentage where an instance was predicted
    as the opposite label (i.e., left-vs-right or high-vs-low).
    """
    # calculate macro-f1
    f1 = f1_score(actual, predicted, average='macro') * 100

    # calculate accuracy
    accuracy = accuracy_score(actual, predicted) * 100

    # calculate the flip error rate
    flip_err = sum([1 for i in range(len(actual)) if abs(actual[i] - predicted[i]) > 1]) / len(actual) * 100

    # calculate mean absolute error (mae)
    mae = sum([abs(actual[i] - predicted[i]) for i in range(len(actual))]) / len(actual)
    mae = mae[0] if not isinstance(mae, float) else mae

    return f1, accuracy, flip_err, mae

def train_model(clf, node2vec_model, data_year='2020', num_labels=3):
    label2int = {
        "fact": {"low": 0, "mixed": 1, "high": 2},
        "bias": {"left": 0, "center": 1, "right": 2},
    }

    # int2label = {
    #     "fact": {0: "low", 1: "mixed", 2: "high"},
    #     "bias": {0: "left", 1: "center", 2: "right"},
    # }

    if data_year == '2020':
        DATA = load_corpus('new_corpus_2020.csv')
        SPLITS = load_splits('modified_splits_new_corpus_2020.json')
    elif data_year == '2018':
        DATA = load_corpus('modified_corpus_2018.csv')
        LOADED_SPLITS = load_splits('splits_2018.json')

        with open(_INVALID_URLS_2018, 'r') as f:
            invalid_urls_2018 = json.load(f)

        SPLITS = []
        for index in range(len(LOADED_SPLITS)):

            train_index_urls = [invalid_urls_2018[train_url] if train_url in invalid_urls_2018 else train_url
                                for train_url in LOADED_SPLITS[index][f"train-{index}"].split('\n')]

            test_index_urls = [invalid_urls_2018[test_url] if test_url in invalid_urls_2018 else test_url
                               for test_url in LOADED_SPLITS[index][f"test-{index}"].split('\n')]

            SPLITS.append({f'test-{index}': '\n'.join(test_index_urls),
                           f'train-{index}': '\n'.join(train_index_urls)})

    else:
        raise ValueError(f'Incorrect parameter "data_year" = {data_year}')

    df = pd.DataFrame(DATA)

    df['source_url_processed'] = df['source_url_processed'].apply(lambda x: 'hemaven.net' if x == 'themaven.net' else x)
    num_folds = len(SPLITS)
    print(f"Splits: {type(SPLITS)} len: {len(SPLITS)}")

    # create a dictionary: the keys are the media and the values are their corresponding labels (transformed to int)
    labels = {df["source_url_processed"].str.lower()[i]: label2int['fact'][df['fact'].str.lower()[i]] for i in range(df.shape[0])}

    # create placeholders where predictions will be cumulated over the different folds
    all_urls = []
    actual = np.zeros(len(df), dtype=np.int)
    predicted = np.zeros(len(df), dtype=np.int)
    probs = np.zeros((len(df), num_labels), dtype=np.float)

    i = 0

    print("Start training...")

    for index in range(num_folds):
        print(f"Fold: {index}")

        # get the training and testing media for the current fold
        urls = {
            "train": SPLITS[index][f"train-{index}"].split('\n'),
            "test": SPLITS[index][f"test-{index}"].split('\n'),
        }

        all_urls.extend(SPLITS[index][f"test-{index}"].split('\n'))

        # initialize the features and labels matrices
        X, y = {}, {}

        # concatenate the different features/labels for the training sources
        X["train"] = np.asmatrix([node2vec_model.wv[url] for url in urls["train"]]).astype("float")
        y["train"] = np.array([labels[url] for url in urls["train"]], dtype=np.int)

        # concatenate the different features/labels for the testing sources
        X["test"] = np.asmatrix([node2vec_model.wv[url] for url in urls["test"]]).astype("float")
        y["test"] = np.array([labels[url] for url in urls["test"]], dtype=np.int)

        # train the classifier
        clf.fit(X["train"], y["train"])

        # generate predictions
        pred = clf.predict(X["test"])

        # generate probabilites
        prob = clf.predict_proba(X["test"])

        # cumulate the actual and predicted labels, and the probabilities over the different folds.  then, move the index
        actual[i: i + y["test"].shape[0]] = y["test"]
        predicted[i: i + y["test"].shape[0]] = pred
        probs[i: i + y["test"].shape[0], :] = prob
        i += y["test"].shape[0]

    # calculate the performance metrics on the whole set of predictions (5 folds all together)
    results = calculate_metrics(actual, predicted)

    # display the performance metrics
    print(f"Accuracy: {results[1]}")
    print(f"Macro-F1: {results[0]}")
    print(f"Flip Error-rate: {results[2]}")
    print(f"MAE: {results[3]}")

    # # map the actual and predicted labels to their categorical format
    # predicted = np.array([int2label[args.task][int(l)] for l in predicted])
    # actual = np.array([int2label[args.task][int(l)] for l in actual])

    # # create a dictionary: the keys are the media, and the values are their actual and predicted labels
    # predictions = {all_urls[i]: (actual[i], predicted[i]) for i in range(len(all_urls))}

    # # create a dataframe that contains the list of m actual labels, the predictions with probabilities.  then store it in the output directory
    # df_out = pd.DataFrame({"source_url": all_urls, "actual": actual, "predicted": predicted, int2label[args.task][0]: probs[:, 0], int2label[args.task][1]: probs[:, 1], int2label[args.task][2]: probs[:, 2],})
    # columns = ["source_url", "actual", "predicted"] + [int2label[args.task][i] for i in range(args.num_labels)]
    # df_out.to_csv(os.path.join(out_dir, "predictions.tsv"), index=False, columns=columns)

    # # write the experiment results in a tabular format
    # res = PrettyTable()
    # res.field_names = ["Macro-F1", "Accuracy", "Flip error-rate", "MAE"]
    # res.add_row(results)

    # # write the experiment summary and outcome into a text file and save it to the output directory
    # with open(os.path.join(out_dir, "results.txt"), "w") as f:
    #     f.write(summary.get_string(title="Experiment Summary") + "\n")
    #     f.write(res.get_string(title="Results"))


def create_graph(lvl_data, root):
    edges = []
    for k in lvl_data[root].keys():
        edges.append((root, k))
        for overlap_site in lvl_data[root][k]['score']:
            edges.append((k, overlap_site['url']))

    return edges


def draw_graph(edges=None, graph=None):
    plt.figure(num=None, figsize=(30, 28), dpi=50)

    if graph:
        nx.draw_networkx(graph.to_networkx())
    else:
        nx.draw_networkx(StellarGraph(edges=edges).to_networkx())


def load_level_data(data_path=None, level=0):
    if not data_path:
        data_path = os.path.join(_DATA_PATH, 'clean_data_20200803.json')

    with open(data_path, 'r') as f:
        data = json.load(f)

    output = {record['sites']: record for record in data if record['levels'] <= level}
    print(f"Loaded {len(output)} nodes with records level <= {level} and child size:{sum([len(record['overlap_sites']) for record in output.values()])}")

    return output


def save_note2vec_model(model):
    pass


def load_corpus(corpus_file):
    corpus = []
    with open(os.path.join(_ANNOTATED_DATA_PATH, corpus_file)) as f:
        csv_reader = csv.DictReader(f)

        for row in csv_reader:
            corpus.append(row)

    return corpus


def create_nodes(lvl_data):
    nodes = []
    for k in list(lvl_data.keys()):
        if not lvl_data[k]['overlap_sites']:
            nodes.append((k, k))
        else:
            for urls in lvl_data[k]['overlap_sites']:
                nodes.append((k, urls['url']))
    return nodes


def create_weighted_nodes(lvl_data):
    nodes = []
    for k in list(lvl_data.keys()):
        if not lvl_data[k]['overlap_sites']:
            nodes.append((k, k, 0.5))
        else:
            for urls in lvl_data[k]['overlap_sites']:
                nodes.append((k, urls['url'], urls.get('overlap_score', 1)))

    return nodes


def get_referral_sites(site, all_data_dir):
    file_name = os.path.join(all_data_dir, f"{site}.html")

    if not os.path.exists(file_name):
        return

    with open(file_name) as f:
        text = f.read()

    text = BeautifulSoup(text, 'html')

    referral_sites = text.find('div', {'id': 'card_referralsites'})

    if not referral_sites:
        return []

    found_sites = referral_sites.find_all('div', {'class': 'Row'})

    return [(found_site.find('div', {'class', 'site'}).a['href'].split('/')[-1],
            found_site.find('span', {'class': 'truncation'}).text.strip())
            for found_site in found_sites]


def get_site_metrics(site, all_data_dir=_ALL_DATA):
    file_name = os.path.join(all_data_dir, f"{site}.html")

    if not os.path.exists(file_name):
        return

    with open(file_name) as f:
        text = f.read()

    text = BeautifulSoup(text, 'html')

    # Site Metrics
    card_metrics = text.find('div', {'id': 'card_metrics'})

    if not card_metrics:
        return []

    engagement_section = card_metrics.find('div', {'class': 'flex'})
    stats = engagement_section.find_all('p', {'class': 'small data'})

    result = {}
    fields = ['Daily Pageviews per Visitor', 'Daily Time on Site', 'Bounce rate']
    for field, stat in zip(fields, stats):
        stat = stat.text.replace("\n\t\t\t\t\t\t\t\t\t          \t            ", '').split(' ')[0]
        result[field] = stat

    # Alexa rank
    card_rank = text.find('div', {'id': 'card_rank'})

    if not card_rank:
        pass

    alexa_rank = {}
    alexa_rank['alexa_rank'] = card_rank.find('p', {'class': 'big data'}).text.replace('\n\t\t\t\t\t\t\t\t\t          \t        ', '')
    alexa_rank['start_rank'] = card_rank.find('div', {'class', 'start-rank'}).text#.find('span', {'class', 'rank'}).text
    alexa_rank['end_rank'] = card_rank.find('div', {'class', 'end-rank'}).text#.find('span', {'class', 'rank'}).text

    print(alexa_rank)

    return result
