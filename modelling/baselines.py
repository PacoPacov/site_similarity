import logging
from collections import Counter, defaultdict

from tqdm import tqdm
from dataprep.alexa_scrapper import ScrapeAlexa
from dataprep.load_annotated_data import apply_splits, load_corpus

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


def baseline_one(data):
    """
    When we have equal or no information about relative sites: We take the most frequent one from the data
    :param data:
    :return:
    """
    most_frequent_bias = list(Counter([_['fact'] for _ in data]).keys())[0]

    result = {}
    for row in tqdm(data):
        try:
            alexa_results = ScrapeAlexa(row['source_url_processed']).scrape_alexa_site_info()
        except BaseException as e:
            _LOGGER.error(f"alexa_rank fails on site: {row['source_url_processed']} with error {repr(e)}")

        if not alexa_results['score']:
            _LOGGER.info(f"Could not find results for: {row['source_url_processed']}")
            result[row['source_url_processed']] = most_frequent_bias
            continue

        annotations = _process_related_sites(alexa_results, data)
        # _LOGGER.info(f"Results for {row['source_url_processed']} {annotations}")

        annotations_counter = Counter([value for value in annotations.values() if value])

        if len(annotations_counter) == 1:
            result[row['source_url_processed']] = list(annotations_counter.keys())[0]
        elif annotations_counter and list(annotations_counter.values())[0] > list(annotations_counter.values())[1]:
            result[row['source_url_processed']] = list(annotations_counter.keys())[0]
        else:
            result[row['source_url_processed']] = most_frequent_bias

    return result


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
    DATA = load_corpus()
    SPLITS = apply_splits(DATA)

    print(SPLITS.keys())

    result_baseline_one = baseline_one(SPLITS['train-0'])

    print(result_baseline_one)

    eval_model(SPLITS['train-0'], result_baseline_one)