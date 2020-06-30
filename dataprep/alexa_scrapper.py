import argparse
import json
import os
import logging

import requests
from bs4 import BeautifulSoup

_DATA_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')
_LOGGER = logging.getLogger('dataprep.alexa_scrapper')


class ScrapeAlexa:
    def __init__(self, target_site):
        self.target_site = target_site

    def _is_site_already_checked(self):
        return f'{self.target_site}.html' in os.listdir(_DATA_PATH)

    @staticmethod
    def _get_alexa_audience_metrics(element):
        """ Returns data about audience overlap like:
        similar sites, overlap score, alexa rank
        :param element: BeautifulSoup object
        """

        similar_sites_by_audience_overlap = element.select("#card_mini_audience")
        if not similar_sites_by_audience_overlap:
            return []

        referral_sites = similar_sites_by_audience_overlap[0]

        score = [score.text for score in referral_sites.find_all('span', {'class', 'truncation'})]
        urls = referral_sites.find_all('div', {'class': 'site'})[1:]

        alexa_ranks = [el.span.text for el in element.find_all('div', {'class': 'metric_two'})[1:6]]

        # TODO Add implementation on how to handle when where isn't 'referral_sites'
        result = []
        for url, score, alexa_rank in zip(urls, score, alexa_ranks):
            url = url.text.strip('\n').strip('\t').strip(' ').strip('\t').strip('\n').strip(' ').strip('\t').strip('\n')
            score = float(score.replace(',', '')) if score.strip() != '-' else None
            alexa_rank = float(alexa_rank[:-2].replace(',', '')) if alexa_rank.strip() != '-' else None
            result.append({'url': url, 'score': score, 'alexa_rank': alexa_rank})

        return result

    def scrape_alexa_site_info(self):
        if self._is_site_already_checked():
            _LOGGER.info(f"This site '{self.target_site}' has already being processed")
            return

        response = requests.get(f"https://www.alexa.com/siteinfo/{self.target_site}")
        with open(os.path.join(_DATA_PATH, f'{self.target_site}.html'), 'w') as f:
            f.write(response.text)
        element = BeautifulSoup(response.text, 'lxml')

        countries = element.find_all('div', {'id': 'countryName'})
        percentages = element.find_all('div', {'id': 'countryPercent'})

        res = {}

        res['site'] = self.target_site
        res['score'] = ScrapeAlexa._get_alexa_audience_metrics(element)
        res['audience_geography'] = [{'country': country.text.split("\xa0")[-1], 'percent': float(percent.text[:-1])}
                                      for country, percent in zip(countries, percentages)]

        return res


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
