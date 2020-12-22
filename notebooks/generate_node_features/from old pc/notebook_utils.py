import json
import os

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import redis
from bs4 import BeautifulSoup
from gensim.models import Word2Vec
from sklearn import svm
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score, mean_absolute_error
from stellargraph import StellarGraph
from stellargraph.data import BiasedRandomWalk

from dataprep.alexa_scrapper import ScrapeAlexa
from dataprep.scrape_all_alexa_information import main
from dataprep.load_annotated_data import load_corpus, load_splits

_DATA_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data')
_ALL_DATA = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'dataset', 'all_data')
_MODEL_STORAGE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data', 'node2vec_models')


def evaluate(y_test, y_test_pred):
    return {
        "Accuracy": accuracy_score(y_test, y_test_pred),
        "Balanced Accuracy score": balanced_accuracy_score(y_test, y_test_pred),
        "F1 micro score": f1_score(y_test, y_test_pred, average='micro'),
        "F1 macro score": f1_score(y_test, y_test_pred, average='macro'),
        "F1 weighted score": f1_score(y_test, y_test_pred, average='weighted'),
        "MAE": mean_absolute_error(y_test, y_test_pred),
        "Confusion matrix": confusion_matrix(y_test, y_test_pred).tolist()
    }


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


def train_model(clf, data_year='2020', node2vec_model=None, num_labels=3):
    label2int = {
        "fact": {"low": 0, "mixed": 1, "high": 2},
        "bias": {"left": 0, "center": 1, "right": 2},
    }

    if data_year == '2020':
        DATA = load_corpus('new_corpus_2020.csv', data_year='2020')
        SPLITS = load_splits('modified_splits_new_corpus_2020.json', data_year='2020')
    elif data_year == '2018':
        DATA = load_corpus('corpus_2018_20200907.tsv', data_year='2018', delimiter='\t')
        SPLITS = load_splits('modified_split_2018_20200907.json', data_year='2018')
    else:
        raise ValueError(f'Incorrect parameter "data_year" = {data_year}')

    df = pd.DataFrame(DATA)

    df['source_url_processed'] = df['source_url_processed'].apply(lambda x: 'hemaven.net' if x == 'themaven.net' else x)
    num_folds = len(SPLITS)

    # create a dictionary: the keys are the media and the values are their corresponding labels (transformed to int)
    labels = {df["source_url_processed"].str.lower()[i]: label2int['fact'][df['fact'].str.lower()[i]]
              for i in range(df.shape[0])}

    # create placeholders where predictions will be cumulated over the different folds
    all_urls = []
    actual = np.zeros(len(df), dtype=np.int)
    predicted = np.zeros(len(df), dtype=np.int)

    i = 0

    print("Start training...")

    for index in range(num_folds):
        # get the training and testing media for the current fold
        urls = {
            "train": SPLITS[index][f"train-{index}"].split('\n'),
            "test": SPLITS[index][f"test-{index}"].split('\n'),
        }

        all_urls.extend(SPLITS[index][f"test-{index}"].split('\n'))

        # initialize the features and labels matrices
        X, y = {}, {}

        # concatenate the different features/labels for the training sources
        if node2vec_model:
            X["train"] = np.asmatrix([node2vec_model.wv[url] for url in urls["train"]]).astype("float")
            X["test"] = np.asmatrix([node2vec_model.wv[url] for url in set(urls["test"])]).astype("float")
        else:
            X["train"] = np.array([url for url in urls["train"]])
            X["test"] = np.array([url for url in set(urls["test"])])

        y["train"] = np.array([labels[url] for url in urls["train"]], dtype=np.int)
        # concatenate the different features/labels for the testing sources
        y["test"] = np.array([labels[url] for url in set(urls["test"])], dtype=np.int)

        # return X, y
        # train the classifier
        clf.fit(X["train"], y["train"])

        # generate predictions
        pred = clf.predict(X["test"])

        # cumulate the actual and predicted labels, and the probabilities over the different folds. then, move the index
        actual[i: i + len(y["test"])] = y["test"]

        predicted[i: i + len(y["test"])] = pred
        i += y["test"].shape[0]

    # calculate the performance metrics on the whole set of predictions (5 folds all together)
    return evaluate(actual, predicted)


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
    print((f"Loaded {len(output)} nodes with records level <= {level} and child size:"
           f"{sum([len(record['overlap_sites']) for record in output.values()])}"))

    return output


def create_node2vec_model(lvl_data, is_weighted, dimension=None, file_name=None, prefix=None, dimensions=[]):
    """Creates a node2vec model and saves it.
    :param lvl_data: data that will be used to create the model
    :dimension: Integer value that tells in which dimension should the embeddings be
    :is_weighted: Boolean value that indicates whenever the graph is weighted or not
    :file_name: Name of the file where the model will be saved. Please use the file extention '.model'
    """
    # TODO Add more checks for the input parameters
    if not dimensions:
        dimensions = [dimension]

    if not file_name:
        weight = 'unweighted' if not is_weighted else 'weight'
        file_names = [f"{prefix}_{weight}_{dimension}D.model" for dimension in dimensions]
    else:
        file_names = [file_name]

    columns = ['source', 'target', 'weight'] if is_weighted else ['source', 'target']

    lvl_one_graph = StellarGraph(edges=pd.DataFrame(lvl_data, columns=columns))

    rw = BiasedRandomWalk(lvl_one_graph)

    print("Start creating random walks")
    walks = rw.run(
        nodes=list(lvl_one_graph.nodes()),  # root nodes
        length=100,  # maximum length of a random walk
        n=10,  # number of random walks per root node
        p=0.5,  # Defines (unnormalized) probability, 1/p, of returning to source node
        q=2.0,  # Defines (unnormalized) probability, 1/q, for moving away from source node
        weighted=is_weighted,  # for weighted random walks
        seed=42,  # random seed fixed for reproducibility
    )
    print("Number of random walks: {}".format(len(walks)))

    str_walks = [[str(n) for n in walk] for walk in walks]

    for d, model_name in zip(dimensions, file_names):
        model = Word2Vec(str_walks, size=d, window=5, min_count=0, sg=1, workers=2, iter=1)

        os.makedirs(_MODEL_STORAGE, exist_ok=True)

        if model_name in os.listdir(_MODEL_STORAGE):
            raise ValueError(f'Model {model_name} already exists in {_MODEL_STORAGE}!')

        model.save(os.path.join(_MODEL_STORAGE, model_name))

        print(f"Successful save of model: {model_name}!")


def load_node2vec_model(model_name):
    return Word2Vec.load(os.path.join(_MODEL_STORAGE, model_name))


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


def get_referral_sites(site):
    redis_client = redis.Redis()

    if redis_client.exists(site):
        return json.loads(redis_client.get(site))

    else:
        alexa_scrapper = ScrapeAlexa()
        content = alexa_scrapper.get_site_content(site)
        referral_sites = alexa_scrapper._get_alexa_referral_sites_metric(content)

        redis_client.set(site, json.dumps(referral_sites))

        return referral_sites


def get_site_metrics(site, all_data_dir=_ALL_DATA):
    file_name = os.path.join(all_data_dir, f"{site}.html")

    if not os.path.exists(file_name):
        return

    with open(file_name) as f:
        html_text = f.read()

    text = BeautifulSoup(html_text, 'html')

    # Site Metrics
    card_metrics = text.find('div', {'id': 'card_metrics'})

    if not card_metrics:
        return []

    engagement_section = card_metrics.find('div', {'class': 'flex'})
    stats = engagement_section.find_all('p', {'class': 'small data'})

    result = {}
    fields = ['Daily Pageviews per Visitor', 'Daily Time on Site', 'Bounce rate']
    for field, stat in zip(fields, stats):
        stat = stat.text.strip(' \t\n').split(' ')[0]
        result[field] = stat

    # Alexa rank
    card_rank = text.find('div', {'id': 'card_rank'})

    if not card_rank:
        pass

    alexa_rank = {}
    alexa_rank['alexa_rank'] = card_rank.find('p', {'class': 'big data'}).text.strip(' \t\n')
    alexa_rank['alexa_rank_in_past_three_months'] = text.find('script', {'id': 'rankData'}).string
    alexa_rank['total_sites_linked_in'] = card_rank.find('span', {'class': 'big data'}).text

    print(alexa_rank)

    return result


def eval_node2vec_models(models, data_year):
    result_report = []

    for model in models:
        print(f'Using model: {model}')
        node2vec_model = load_node2vec_model(model)

        clf = LogisticRegressionCV(Cs=10, cv=5, scoring="accuracy", multi_class="ovr", max_iter=300, random_state=42)
        result_report.append([
            model.strip('.model'),
            'LogisticRegression CV = 5',
            *list(train_model(clf, node2vec_model=node2vec_model, data_year=data_year).values())
        ])

        clf2 = LogisticRegressionCV(Cs=10, cv=10, scoring="accuracy", multi_class="ovr", max_iter=300, random_state=42)
        result_report.append([
            model.strip('.model'),
            'LogisticRegression CV = 10',
            *list(train_model(clf2, node2vec_model=node2vec_model, data_year=data_year).values())
        ])

        tree_clf = GradientBoostingClassifier(random_state=42)
        result_report.append([
            model.strip('.model'),
            'GradientBoostingClassifier',
            *list(train_model(tree_clf, node2vec_model=node2vec_model, data_year=data_year).values())
        ])

        svm_clf = svm.SVC(decision_function_shape='ovo', probability=True, random_state=42)
        result_report.append([
            model.strip('.model'),
            'SVC ovo',
            *list(train_model(svm_clf, node2vec_model=node2vec_model, data_year=data_year).values())
        ])

    return pd.DataFrame(result_report,
                        columns=["Feature", "Classifier", "Accuracy", "Balanced Accuracy score",
                                 "F1 micro score", "F1 macro score", "F1 weighted score", "MAE", "Confusion matrix"])


def load_json(path):
    with open(path) as f:
        data = json.load(f)

    return data


def get_referral_sites_edges(data):
    nodes = []

    for base_url, referral_sites in data.items():
        if not referral_sites:
            nodes.append((base_url, base_url))
        else:
            for referral_site, _ in referral_sites:
                if referral_site != base_url:
                    nodes.append((base_url, referral_site))

    print('Node length:', len(nodes))
    print('Distinct node length:', len(set(nodes)))

    return set(nodes)


def get_alexa_information_sections(target_sites, specific_section='site_metrics'):
    '''
    Note this function should be run after the scrapping is done
    '''
    r = redis.Redis()

    target_data = None

    for site in target_sites:
        if r.get(site):
            target_data = json.loads(r.get(site))
        else:
            target_data = main(site)
            r.set(site, json.dumps(target_data))

        yield target_data[specific_section] if specific_section else target_data


def combined_nodes_referral_sites_audience_overlap(data_year='2020', level=1):
    if data_year == '2018':
        referral_sites_files = [
            'modified_corpus_2018_referral_sites.json',
            'modified_corpus_2018_referral_sites_level_1.json',
            'modified_corpus_2018_referral_sites_level_2.json',
            'modified_corpus_2018_referral_sites_level_3.json'
        ]

        audience_overlap_scrapping_file = 'corpus_2018_audience_overlap_sites_scrapping_result.json'
    elif data_year == '2020':
        referral_sites_files = [
            'corpus_2020_referral_sites.json',
            'corpus_2020_referral_sites_level_1.json',
            'corpus_2020_referral_sites_level_2.json',
            'corpus_2020_referral_sites_level_3.json',
        ]

        audience_overlap_scrapping_file = 'corpus_2020_audience_overlap_sites_scrapping_result.json'
    else:
        raise ValueError('Incorrect argument "data_year" should be "2018" or "2020"!')

    referral_sites = {}

    for f in referral_sites_files[:level + 1]:
        loaded_data = load_json(os.path.join(_DATA_PATH, f))
        print(f'For file "{f}" -> load {len(loaded_data)} records')
        referral_sites.update(loaded_data)

    referral_sites_NODES = []

    for base_url, referral_sites in referral_sites.items():
        if not referral_sites:
            referral_sites_NODES.append((base_url, base_url))

        for referral_site, _ in referral_sites:
            if referral_site != base_url:
                referral_sites_NODES.append((base_url, referral_site))

    audience_overlap_sites = load_level_data(os.path.join(_DATA_PATH, audience_overlap_scrapping_file), level=level)

    audience_overlap_sites_NODES = create_nodes(audience_overlap_sites)

    print('referral_sites node size:', len(referral_sites_NODES),
          'audience_overlap node size:', len(audience_overlap_sites_NODES))

    return audience_overlap_sites_NODES + referral_sites_NODES


def extract_node_features(res):
    df_indexs = []
    alexa_ranks = []
    daily_pageviews_per_visitors = []
    daily_time_on_sites = []
    total_sites_linking_ins = []
    bounce_rates = []

    for site_name, site_info in res.items():
        alexa_rank = int(site_info['alexa_rank']['site_rank'].replace(',', '').strip(' #')) if site_info['alexa_rank'].get('site_rank') else None

        df_indexs.append(site_name)
        alexa_ranks.append(alexa_rank)

        if site_info['site_metrics']:
            daily_pageviews_per_visitor = float(site_info['site_metrics']['daily_pageviews_per_visitor']) if site_info['site_metrics']['daily_pageviews_per_visitor'] else None
            
            if site_info['site_metrics']['daily_time_on_site']:
                minutes, seconds = site_info['site_metrics']['daily_time_on_site'].split(':')
                daily_time_on_site = int(minutes) * 60 + int(seconds)
            else:
                daily_time_on_site = None

            total_sites_linking_in = int(site_info['site_metrics']['total_sites_linking_in'].replace(',', '')) if site_info['site_metrics']['total_sites_linking_in'] else None

            bounce_rate = float(site_info['site_metrics']['bounce_rate'].strip('%')) / 100 if site_info['site_metrics']['bounce_rate'] else None

            daily_pageviews_per_visitors.append(daily_pageviews_per_visitor)
            daily_time_on_sites.append(daily_time_on_site)
            total_sites_linking_ins.append(total_sites_linking_in)
            bounce_rates.append(bounce_rate)
        else:
            daily_pageviews_per_visitors.append(None)
            daily_time_on_sites.append(None)
            total_sites_linking_ins.append(None)
            bounce_rates.append(None)

    return df_indexs, {
        'alexa_ranks': alexa_ranks,
        'daily_pageviews_per_visitors': daily_pageviews_per_visitors,
        'daily_time_on_sites': daily_time_on_sites,
        'total_sites_linking_ins': total_sites_linking_ins, 
        'bounce_rate': bounce_rates}