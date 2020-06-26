import argparse
import json
import os

import requests
from bs4 import BeautifulSoup

_DATA_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')


class ScrapeAlexa():
    @staticmethod
    def _get_alexa_audience_metrics(element):
        """ Returns data about audience overlap like:
        similar sites, overlap score, alexa rank
        :param element: BeautifulSoup object
        """
        referral_sites = element.select("#card_mini_audience")[0]

        score = [score.text for score in referral_sites.find_all('span', {'class', 'truncation'})]
        urls = referral_sites.find_all('div', {'class': 'site'})[1:]

        alexa_ranks = [el.span.text for el in element.find_all('div', {'class': 'metric_two'})[1:6]]

        result = []
        for url, score, alexa_rank in zip(urls, score, alexa_ranks):
            url = url.text.strip('\n').strip('\t').strip(' ').strip('\t').strip('\n').strip(' ').strip('\t').strip('\n')
            score = float(score.replace(',', ''))
            alexa_rank = float(alexa_rank[:-2].replace(',', ''))
            result.append({'url': url, 'score': score, 'alexa_rank': alexa_rank})

        return result

    def scrape_alexa_site_info(self, target_site):
        response = requests.get(f"https://www.alexa.com/siteinfo/{target_site}")
        with open(os.path.join(_DATA_PATH, f'{target_site}.html'), 'w') as f:
            f.write(response.text)
        element = BeautifulSoup(response.text, 'lxml')

        countries = element.find_all('div', {'id': 'countryName'})
        percentages = element.find_all('div', {'id': 'countryPercent'})

        res = {}

        res['site'] = target_site
        res['score'] = ScrapeAlexa._get_alexa_audience_metrics(element)
        res['audience_geography'] = [{'country': country.text.split("\xa0")[-1], 'percent': float(percent.text[:-1])}
                                      for country, percent in zip(countries, percentages)]

        return res


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--target_site', default='bradva.bg')

    args = parser.parse_args()

    result = ScrapeAlexa().scrape_alexa_site_info(args.target_site)

    with open(os.path.join(_DATA_PATH, f'{args.target_site}.json'), 'w') as f:
        json.dump(result, f, indent=4)

"""
Example usage:
    >>> python site_similarity/dataprep/alexa_scrapper.py --target_site "bradva.bg" 
"""
