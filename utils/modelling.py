import json
import os

import numpy as np
import pandas as pd
from gensim.models import Word2Vec
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                             confusion_matrix, f1_score, mean_absolute_error)
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC
from stellargraph import StellarGraph
from stellargraph.data import BiasedRandomWalk

from dataprep.load_annotated_data import load_corpus, load_splits
from utils.notebook_utils import _DATA_PATH

np.random.seed(16)

_PROJECT_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_MODEL_STORAGE = os.path.join(_PROJECT_PATH, 'data', 'node2vec_models')
_FEATURES_DIR = os.path.join(_DATA_PATH, 'features')
_PARAMS_SVM = [dict(kernel=["rbf"], gamma=np.logspace(-6, 1, 8), C=np.logspace(-2, 2, 5))]

def load_node2vec_model(model_name):
    return Word2Vec.load(os.path.join(_MODEL_STORAGE, model_name))


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


def eval_node2vec_models(models, data_year, task='fact'):
    result_report = []

    for model in models:
        print(f'Using model: {model}')
        node2vec_model = load_node2vec_model(model)

        clf = LogisticRegressionCV(Cs=10, cv=5, scoring="accuracy", multi_class="ovr", max_iter=300, random_state=42)
        result_report.append([
            model.strip('.model'),
            'LogisticRegression CV = 5',
            *list(train_model(clf, node2vec_model=node2vec_model, data_year=data_year, task=task).values())
        ])

        clf2 = LogisticRegressionCV(Cs=10, cv=10, scoring="accuracy", multi_class="ovr", max_iter=300, random_state=42)
        result_report.append([
            model.strip('.model'),
            'LogisticRegression CV = 10',
            *list(train_model(clf2, node2vec_model=node2vec_model, data_year=data_year, task=task).values())
        ])

        tree_clf = GradientBoostingClassifier(random_state=42)
        result_report.append([
            model.strip('.model'),
            'GradientBoostingClassifier',
            *list(train_model(tree_clf, node2vec_model=node2vec_model, data_year=data_year, task=task).values())
        ])

        svm_clf = SVC(decision_function_shape='ovo', probability=True, random_state=42)
        result_report.append([
            model.strip('.model'),
            'SVC ovo',
            *list(train_model(svm_clf, node2vec_model=node2vec_model, data_year=data_year, task=task).values())
        ])

    return pd.DataFrame(result_report,
                        columns=["Feature", "Classifier", "Accuracy", "Balanced Accuracy score",
                                 "F1 micro score", "F1 macro score", "F1 weighted score", "MAE", "Confusion matrix"])


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


def train_model(clf, data_year='2020', node2vec_model=None, task='fact', num_labels=3):
    label2int = {
        "fact": {"low": 0, "mixed": 1, "high": 2},
        "bias": {"left": 0, 'extreme-left': 0,
                 "center": 1, 'right-center': 1, 'left-center': 1,
                 "right": 2, 'extreme-right': 2},
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

    # df['source_url_processed'] = df['source_url_processed'].apply(lambda x: 'hemaven.net' if x == 'themaven.net' else x)
    num_folds = len(SPLITS)

    # create a dictionary: the keys are the media and the values are their corresponding labels (transformed to int)
    labels = {df["source_url_processed"].str.lower()[i]: label2int[task][df[task].str.lower()[i]]
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


def train_official_model(data_year='2020', node2vec_model=None, normalize_features=False, num_labels=3):
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

        if normalize_features:
            # normalize the features values
            scaler = MinMaxScaler()
            scaler.fit(X["train"])
            X["train"] = scaler.transform(X["train"])
            X["test"] = scaler.transform(X["test"])

        # fine-tune the model
        clf_cv = GridSearchCV(SVC(), scoring="f1_macro", cv=num_folds, n_jobs=4, param_grid=_PARAMS_SVM)
        clf_cv.fit(X["train"], y["train"])

        # train the final classifier using the best parameters during crossvalidation
        clf = SVC(
            kernel=clf_cv.best_estimator_.kernel,
            gamma=clf_cv.best_estimator_.gamma,
            C=clf_cv.best_estimator_.C,
            probability=True
        )
        clf.fit(X["train"], y["train"])

        # generate predictions
        pred = clf.predict(X["test"])

        # cumulate the actual and predicted labels, and the probabilities over the different folds. then, move the index
        actual[i: i + len(y["test"])] = y["test"]

        predicted[i: i + len(y["test"])] = pred
        i += y["test"].shape[0]

    # calculate the performance metrics on the whole set of predictions (5 folds all together)
    return evaluate(actual, predicted)


def export_node2vec_as_feature(model_name, data_year='2020'):
    model = load_node2vec_model(model_name)

    if data_year == '2020':
        url_mapping = None
        corpus = load_corpus('new_corpus_2020.csv', data_year='2020')
    elif data_year == '2018':
        url_mapping = {
            "conservativeoutfitters.com": "conservativeoutfitters.com-blogs-news",
            "who.int": "who.int-en",
            "themaven.net": "themaven.net-beingliberal",
            "al-monitor.com": "al-monitor.com-pulse-home.html",
            "pri.org": "pri.org-programs-globalpost",
            "mlive.com": "mlive.com-grand-rapids-#-0",
            "pacificresearch.org": "pacificresearch.org-home",
            "telesurtv.net": "telesurtv.net-english",
            "elpais.com": "elpais.com-elpais-inenglish.html",
            "inquisitr.com": "inquisitr.com-news",
            "cato.org": "cato.org-regulation",
            "jpost.com": "jpost.com-Jerusalem-Report",
            "newcenturytimes.com": "newcenturytimes.com",
            "oregonlive.com": "oregonlive.com-#-0",
            "rfa.org": "rfa.org-english",
            "people.com": "people.com-politics",
            "russia-insider.com": "russia-insider.com-en",
            "nola.com": "nola.com-#-0",
            "host.madison.com": "host.madison.com-wsj",
            "conservapedia.com": "conservapedia.com-Main_Page",
            "futureinamerica.com": "futureinamerica.com-news",
            "indymedia.org": "indymedia.org-or-index.shtml",
            "newyorker.com": "newyorker.com-humor-borowitz-report",
            "rt.com": "rt.com-news",
            "westernjournalism.com": "westernjournalism.com-thepoint",
            "scripps.ucsd.edu": "scripps.ucsd.edu-news",
            "citizensunited.org": "citizensunited.org-index.aspx",
            "gallup.com": "gallup.com-home.aspx",
            "news.harvard.edu": "news.harvard.edu-gazette",
            "spin.com": "spin.com-death-and-taxes",
            "itv.com": "itv.com-news",
            "theguardian.com": "theguardian.com-observer",
            "concernedwomen.org": "concernedwomen.org-blog",
            "npr.org": "npr.org-sections-news",
            "yahoo.com": "yahoo.com-news-?ref=gs",
            "zcomm.org": "zcomm.org-zmag",
            "therealnews.com": "therealnews.com-t2"
        }
        corpus = load_corpus('corpus_2018_20200907.tsv', data_year='2018', delimiter='\t')
    else:
        raise ValueError(f'Invalid data_year parameter {data_year}')

    feature = {}
    for record in corpus:
        site = record['source_url_processed']
        model_mapping = url_mapping[site] if url_mapping and site in url_mapping else site

        print(site, model_mapping)
        if data_year == '2018' and site in ['newyorker.com', 'westernjournalism.com', 'pri.org', 'mlive.com']:
            feature[site] = model[site].tolist()
        feature[model_mapping] = model[site].tolist()

    feature_name = os.path.join(_FEATURES_DIR, model_name.strip('.model') + ".json")
    with open(feature_name, 'w') as f:
        json.dump(feature, f)
