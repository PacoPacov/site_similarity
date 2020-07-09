import logging
from collections import Counter, defaultdict

from tqdm import tqdm
from dataprep.alexa_scrapper import ScrapeAlexa

_LOGGER = logging.getLogger('modelling.baselines')


def eval_model(data, predicted_data):
    correct, incorrect = 0, 0
    for predicted_url in predicted_data:
        for url in data:
            if predicted_url == url['source_url_processed']:
                if predicted_data[predicted_url] == url['fact']:
                    correct += 1
                else:
                    incorrect += 1

    print(f"Accuracy: {correct / len(predicted_data)}")
    print(f"Precision: {correct / (correct + incorrect)}")


def _process_related_sites(alexa_results, annotated_data):
    annotations = {res['url']: None for res in alexa_results['score']}

    for res in alexa_results['score']:
        for url in annotated_data:
            if res['url'] == url['source_url_processed']:
                annotations[res['url']] = url['fact']

    return annotations


class MostFrequentClassifier:
    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs

    def fit(self, data):
        self.most_frequent_classes_ = list(Counter([_['fact'] for _ in data]).keys())

    def predict(self, data):
        result = {}
        for row in tqdm(data):
            try:
                alexa_results = ScrapeAlexa(row['source_url_processed']).scrape_alexa_site_info()
            except BaseException as e:
                _LOGGER.error(f"alexa_rank fails on site: {row['source_url_processed']} with error {repr(e)}")

            if not alexa_results['score']:
                _LOGGER.info(f"Could not find results for: {row['source_url_processed']}")
                result[row['source_url_processed']] = self.most_frequent_classes_[0]
                continue

            annotations = _process_related_sites(alexa_results, data)

            annotations_counter = Counter([value for value in annotations.values() if value])

            print("URL:", row['source_url_processed'], "Annotations Counter:", annotations_counter)

            if len(annotations_counter) == 1:
                result[row['source_url_processed']] = list(annotations_counter.keys())[0]
            elif annotations_counter and list(annotations_counter.values())[0] > list(annotations_counter.values())[1]:
                result[row['source_url_processed']] = list(annotations_counter.keys())[0]
            elif annotations_counter and list(annotations_counter.values())[0] == list(annotations_counter.values())[1]:
                index0 = self.most_frequent_classes_.index(list(annotations_counter.keys())[0])

                index1 = self.most_frequent_classes_.index(list(annotations_counter.keys())[1])
                if index0 < index1:
                    result[row['source_url_processed']] = list(annotations_counter.keys())[0]
                else:
                    result[row['source_url_processed']] = list(annotations_counter.keys())[1]
            elif annotations_counter and list(annotations_counter.values())[0] < list(annotations_counter.values())[1]:
                result[row['source_url_processed']] = list(annotations_counter.keys())[1]
            else:
                result[row['source_url_processed']] = self.most_frequent_classes_[0]


class OverlapScoreClassifier:
    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs

    def fit(self, data):
        self.most_frequent_classes_ = list(Counter([_['fact'] for _ in data]).keys())

    def predict(self, data):
        result = {}
        for row in tqdm(data):
            try:
                alexa_results = ScrapeAlexa(row['source_url_processed']).scrape_alexa_site_info()
            except BaseException as e:
                _LOGGER.error(f"alexa_rank fails on site: {row['source_url_processed']} with error {repr(e)}")

            if not alexa_results['score']:
                _LOGGER.info(f"Could not find results for: {row['source_url_processed']}")
                result[row['source_url_processed']] = self.most_frequent_classes_[0]
                continue

            annotations = _process_related_sites(alexa_results, data)

            annotations_counter = Counter([value for value in annotations.values() if value])

            print("URL:", row['source_url_processed'], "Annotations Counter:", annotations_counter)

            if len(annotations_counter) == 1:
                result[row['source_url_processed']] = list(annotations_counter.keys())[0]
            elif annotations_counter and list(annotations_counter.values())[0] > list(annotations_counter.values())[1]:
                result[row['source_url_processed']] = list(annotations_counter.keys())[0]
            elif annotations_counter and list(annotations_counter.values())[0] == list(annotations_counter.values())[1]:
                index0 = self.most_frequent_classes_.index(list(annotations_counter.keys())[0])

                index1 = self.most_frequent_classes_.index(list(annotations_counter.keys())[1])
                if index0 < index1:
                    result[row['source_url_processed']] = list(annotations_counter.keys())[0]
                else:
                    result[row['source_url_processed']] = list(annotations_counter.keys())[1]
            elif annotations_counter and list(annotations_counter.values())[0] < list(annotations_counter.values())[1]:
                result[row['source_url_processed']] = list(annotations_counter.keys())[1]
            else:
                result[row['source_url_processed']] = self.most_frequent_classes_[0]


def baseline_two(data):
    """
    We look at overlap score and take the label of this site
    :param data:
    :return:
    """
    # TODO implement the baseline in more ML way
    pass


def baseline_three(data):
    """
    We don't look at votes we take the label from the first annotated related site.
    :param data:
    :return:
    """
    # TODO implement the baseline in more ML way
    pass


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
