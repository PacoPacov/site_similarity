import argparse
import copy
import json
import os
import logging
import time

import requests
from bs4 import BeautifulSoup
from tqdm import tqdm

_DATA_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')
_LOGGER = logging.getLogger('dataprep.alexa_scrapper')


class ScrapeAlexa:
    def __init__(self, target_site, target_dir=None):
        self.target_site = target_site
        self.target_dir = target_dir if target_dir else _DATA_PATH
        os.makedirs(self.target_dir, exist_ok=True)

    def _is_site_already_checked(self):
        return f'{self.target_site}.html' in os.listdir(self.target_dir)

    @staticmethod
    def _get_alexa_audience_metrics(element):
        """ Returns data about audience overlap like:
        similar sites, overlap score, alexa rank
        :param element: BeautifulSoup object
        """

        similar_sites_by_audience_overlap = element.find('div', {'id': 'card_mini_audience'})
        audience_overlap = element.find('div', {'id': 'card_overlap'})

        if similar_sites_by_audience_overlap:
            resources = similar_sites_by_audience_overlap
        elif audience_overlap:
            resources = audience_overlap
        else:
            return

        similar_sites = [el['href'] for el in resources.find_all('a', {'class': 'truncation'})]
        overlap_score = [el.text for el in resources.find_all('span', {'class': 'truncation'})]

        sites_overlap_score = overlap_score[::2]
        alexa_score = overlap_score[1::2]

        # TODO Add implementation on how to handle when where isn't 'referral_sites'

        return [{'url': site.replace("/siteinfo/", ''),
                 'overlap_score': float(ov_score) if ov_score.strip() != '-' else None,
                 'alexa_rank': float(al_score.replace(',', '')) if al_score.strip() != '-' else None}
                for site, ov_score, al_score in zip(similar_sites, sites_overlap_score, alexa_score)]

    def scrape_alexa_site_info(self):
        if self._is_site_already_checked():
            _LOGGER.info(f"This site '{self.target_site}' has already being processed")
            with open(f"{self.target_dir}/{self.target_site}.html") as f:
                response = f.read()

            element = BeautifulSoup(response, 'lxml')

        else:
            time.sleep(10)
            response = requests.get(f"https://www.alexa.com/siteinfo/{self.target_site}")

            if 'Forbidden' in response.text:
                raise ValueError("You've exceeded the request limit for the day!")

            with open(os.path.join(self.target_dir, f'{self.target_site}.html'), 'w') as f:
                f.write(response.text)
            element = BeautifulSoup(response.text, 'lxml')

        countries = element.find_all('div', {'id': 'countryName'})
        percentages = element.find_all('div', {'id': 'countryPercent'})

        res = {}

        res['site'] = self.target_site
        score = ScrapeAlexa._get_alexa_audience_metrics(element)
        if not score:
            _LOGGER.info(f"No similar_sites_by_audience_overlap or audience_overlap found for {self.target_site}")
        res['score'] = score
        if countries:
            res['audience_geography'] = [{'country': country.text.split("\xa0")[-1], 'percent': float(percent.text[:-1])}
                                          for country, percent in zip(countries, percentages)]
        else:
            res['audience_geography'] = []

        return res


def level_one_scrapping(data, target_dir=None):
    level_one_result = copy.deepcopy(data)
    for site in tqdm(data):
        level_one_res = []
        for overlap_site in data[site]['score']:
            level_one_res.append(ScrapeAlexa(overlap_site['url'], target_dir).scrape_alexa_site_info())

        level_one_result[site]['level_one_res'] = level_one_res

    return level_one_result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--target_site', default='bradva.bg')

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    result = ScrapeAlexa(args.target_site).scrape_alexa_site_info()

    with open(os.path.join(_DATA_PATH, f'{args.target_site}.json'), 'w') as f:
        json.dump(result, f, indent=4)

"""
Example usage:
    >>> python site_similarity/dataprep/alexa_scrapper.py --target_site "bradva.bg" 
"""
