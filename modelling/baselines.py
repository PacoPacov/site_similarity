import logging
from abc import abstractmethod
from collections import Counter, defaultdict

from tqdm import tqdm
from dataprep.alexa_scrapper import ScrapeAlexa

_LOGGER = logging.getLogger('modelling.baselines')


class BaseClassifier:
    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs

    def fit(self, data):
        self.most_frequent_classes_ = [label for label, count in Counter([_['fact'] for _ in data]).most_common()]

    @abstractmethod
    def predict(self, data):
        pass


class MostFrequentClassifier(BaseClassifier):
    """
    Classifier that uses the information about most frequent label in the train data to resolve situation
    where we have the same number of labels for the relevant sites.
    """

    @staticmethod
    def _process_alexa_data(alexa_results, annotated_data):
        """
        Will process the returned urls from alexa rank and will check if he has annotations for any of them.
        :param alexa_results: Result from alexa
        :param annotated_data: Annotated data that contains labels
        :return: Dictionary with keys - urls and values the label of the site ['HIGH', 'LOW', 'MIXED'] and None if
        we don't have information about the url.
        """
        annotations = {res['url']: None for res in alexa_results['score']}

        for res in alexa_results['score']:
            for url in annotated_data:
                if res['url'] == url['source_url_processed']:
                    annotations[res['url']] = url['fact']

        return annotations

    def predict(self, data):
        result = {}
        alexa_scrapper = ScrapeAlexa()
        for row in tqdm(data):
            target_url = row['source_url_processed']
            try:
                alexa_results = alexa_scrapper.scrape_alexa_site_info(target_url)
            except BaseException as e:
                _LOGGER.error(f"alexa_rank fails on site: {target_url} with error: {repr(e)}")
                continue

            if not alexa_results['score']:  # no result from alexa
                _LOGGER.info(f"Could not find similar sites for target_site: {target_url}")
                result[target_url] = self.most_frequent_classes_[0]
                continue

            processed_alexa_results = MostFrequentClassifier._process_alexa_data(alexa_results, data)

            annotations = dict(Counter([value for value in processed_alexa_results.values() if value]).most_common())

            if len(annotations) == 1:
                result[target_url] = list(annotations.keys())[0]
            elif annotations and list(annotations.values())[0] > list(annotations.values())[1]:
                result[target_url] = list(annotations.keys())[0]
            elif annotations and list(annotations.values())[0] == list(annotations.values())[1]:
                index0 = self.most_frequent_classes_.index(list(annotations.keys())[0])
                index1 = self.most_frequent_classes_.index(list(annotations.keys())[1])

                if index0 < index1:
                    result[target_url] = list(annotations.keys())[0]
                else:
                    result[target_url] = list(annotations.keys())[1]

            elif annotations and list(annotations.values())[0] < list(annotations.values())[1]:
                result[target_url] = list(annotations.keys())[1]
            else:
                # If there are no annotated rows in the data we return the most frequent one
                result[target_url] = self.most_frequent_classes_[0]

            if annotations:
                print(f"URL: {target_url} Annotations Counter: {annotations} ==> RESULT label: {result[target_url]}")

        return result


class OverlapClassifier(BaseClassifier):
    """
    Classifier that uses the information about the bigger overlap score between target site and the other sites.
    """
    @staticmethod
    def _process_alexa_data(alexa_results, annotated_data):
        """
        Will process the returned urls from alexa rank and will check if he has annotations for any of them.
        :param alexa_results: Result from alexa
        :param annotated_data: Annotated data that contains labels
        :return: Dictionary with keys - urls and values the label of the site ['HIGH', 'LOW', 'MIXED'] and None if
        we don't have information about the url.
        """
        annotations = {res['url']: {} for res in alexa_results['score']}

        for res in alexa_results['score']:
            for url in annotated_data:
                if res['url'] == url['source_url_processed']:
                    annotations[res['url']]['label'] = url['fact']
                    annotations[res['url']]['overlap_score'] = res['overlap_score']

        return {k: v.get('label') for k, v in sorted(annotations.items(),
                                                     key=lambda item: item[1].get('overlap_score', 0),
                                                     reverse=True) if v}

    def predict(self, data):
        result = {}
        alexa_scrapper = ScrapeAlexa()
        for row in tqdm(data):
            target_url = row['source_url_processed']
            try:
                alexa_results = alexa_scrapper.scrape_alexa_site_info(target_url)
            except BaseException as e:
                _LOGGER.error(f"alexa_rank fails on site: {target_url} with error: {repr(e)}")
                continue

            if not alexa_results['score']:  # no result from alexa
                _LOGGER.info(f"Could not find similar sites for target_site: {target_url}")
                result[target_url] = self.most_frequent_classes_[0]
                continue

            annotations = OverlapClassifier._process_alexa_data(alexa_results, data)

            if annotations:
                result[target_url] = list(annotations.values())[0]
                print(f"URL: {target_url} Annotations Counter: {annotations} == > RESULT label: {result[target_url]}")
            else:
                # If there are no annotated rows in the data we return the most frequent one
                result[target_url] = self.most_frequent_classes_[0]

        return result


class FirstAnnotatedSiteClassifier(BaseClassifier):
    """
    We don't look at votes we take the label from the first annotated related site.
    """
    @staticmethod
    def _process_alexa_data(alexa_results, annotated_data):
        for res in alexa_results['score']:
            for url in annotated_data:
                if res['url'] == url['source_url_processed']:
                    return url['fact']

    def predict(self, data):
        result = {}
        alexa_scrapper = ScrapeAlexa()
        for row in tqdm(data):
            target_url = row['source_url_processed']
            try:
                alexa_results = alexa_scrapper.scrape_alexa_site_info(target_url)
            except BaseException as e:
                _LOGGER.error(f"alexa_rank fails on site: {target_url} with error: {repr(e)}")
                continue

            if not alexa_results['score']:  # no result from alexa
                _LOGGER.info(f"Could not find similar sites for target_site: {target_url}")
                result[target_url] = self.most_frequent_classes_[0]
                continue

            annotation = FirstAnnotatedSiteClassifier._process_alexa_data(alexa_results, data)

            if annotation:
                result[target_url] = annotation
            else:
                # If there are no annotated rows in the data we return the most frequent one
                result[target_url] = self.most_frequent_classes_[0]

        return result


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


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
