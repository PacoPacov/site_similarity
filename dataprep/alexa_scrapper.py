import argparse
import json
import logging
import os
import random
import time

import redis
import requests
from bs4 import BeautifulSoup
from tqdm import tqdm

_DATA_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'dataset', 'all_data')
_LOGGER = logging.getLogger('dataprep.alexa_scrapper')


class ScrapeAlexa:
    def __init__(self, target_dir=None):
        self.target_dir = target_dir if target_dir else _DATA_PATH
        os.makedirs(self.target_dir, exist_ok=True)

    def _is_site_already_checked(self):
        return f'{self.target_site}.html' in os.listdir(self.target_dir)

    @staticmethod
    def _get_alexa_audience_metric(content):
        """ Returns data about audience overlap like:
        similar sites, overlap score, alexa rank
        :param content: Page content - HTML string
        """
        element = BeautifulSoup(content, 'lxml')

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

        try:
            parsed_result = [{'url': site.replace("/siteinfo/", ''),
                              'overlap_score': float(ov_score) if ov_score and ov_score.strip() != '-' else None,
                              'alexa_rank': float(al_score.replace(',', '')) if al_score and al_score.strip() != '-' else None}
                             for site, ov_score, al_score in zip(similar_sites, sites_overlap_score, alexa_score)]
        except ValueError as e:
            _LOGGER.error(f'{similar_sites}')
            parsed_result = None

        return parsed_result

    @staticmethod
    def _get_alexa_referral_sites_metric(content):
        """ Returns data about referral like:
        referral sites,
        number of sites linking to target_site that Alexa's web crawl has found.
        :param content: Page content - HTML string
        """

        element = BeautifulSoup(content, 'lxml')
        referral_sites = element.find('div', {'id': 'card_referralsites'})

        if not referral_sites:
            return []

        result = []

        for found_site in referral_sites.find_all('div', {'class': 'Row'}):
            url = found_site.find('div', {'class', 'site'}).a['href'].split('/')[-1]
            score = found_site.find('span', {'class': 'truncation'}).text.strip()

            result.append((url, score))

        return result

    def get_site_content(self, target_site):
        self.target_site = target_site

        if self._is_site_already_checked():
            _LOGGER.info(f"This site '{self.target_site}' has already being processed")
            with open(f"{self.target_dir}/{self.target_site}.html", 'r', encoding='utf-8') as f:
                content = f.read()
        else:
            time.sleep(random.randint(1, 10))
            try:
                response = requests.get(f"https://www.alexa.com/siteinfo/{self.target_site}")
            except ConnectionError as e:
                _LOGGER.error(f"ConnectionError occurred when scrapping site: {self.target_site}. With error: {repr(e)}")
                time.sleep(20)
                response = requests.get(f"https://www.alexa.com/siteinfo/{self.target_site}")
            except Exception as e:
                _LOGGER.error(f"BaseError occurred when scrapping site: {self.target_site}. With error: {repr(e)}")
                time.sleep(20)
                response = requests.get(f"https://www.alexa.com/siteinfo/{self.target_site}")

            if "Alexa is temporarily unavailable.We're working hard to resolve the issue — please try again later" in response.text:
                time.sleep(20)
                response = requests.get(f"https://www.alexa.com/siteinfo/{self.target_site}")

            content = response.text

            with open(os.path.join(self.target_dir, f'{self.target_site}.html'), 'w', encoding='utf-8') as f:
                f.write(content)

        return content

    def scrape_alexa_site_info(self, target_site):
        content = self.get_site_content(target_site)

        element = BeautifulSoup(content, 'lxml')
        countries = element.find_all('div', {'id': 'countryName'})
        percentages = element.find_all('div', {'id': 'countryPercent'})

        res = dict()
        res['site'] = self.target_site
        score = ScrapeAlexa._get_alexa_audience_metric(content)
        if not score:
            _LOGGER.info(f"No similar_sites_by_audience_overlap or audience_overlap found for {self.target_site}")
        res['score'] = score
        if countries:
            res['audience_geography'] = [{'country': country.text.split("\xa0")[-1], 'percent': float(percent.text[:-1])}
                                         for country, percent in zip(countries, percentages)]
        else:
            res['audience_geography'] = []

        return res


def scrapping(data, target_dir=None, output_file=None):
    result = {}
    alexa_scrapper = ScrapeAlexa(target_dir)
    redis_client = redis.Redis(host='localhost', port=6379, db=0)

    for index, child_sites in enumerate(tqdm(data.values())):
        if index % 100 == 0 and output_file:
            print(f"Save data at index {index} ...")
            with open(output_file, 'w') as f:
                json.dump(result, f, indent=4)
        for child_site in child_sites.values():
            result[child_site['site']] = {}
            for overlap_site in child_site.get('score', []):
                if redis_client.exists(overlap_site['url']):
                    # load from redis
                    result[child_site['site']][overlap_site['url']] = json.loads(redis_client.get(overlap_site['url']))
                else:
                    alexa_result = alexa_scrapper.scrape_alexa_site_info(overlap_site['url'])
                    # save in redis
                    redis_client.set(overlap_site['url'], json.dumps(alexa_result))
                    result[child_site['site']][overlap_site['url']] = alexa_result
    return result


def scrapping_refferal_site(data, target_dir=None, output_file=None):
    result = {}
    alexa_scrapper = ScrapeAlexa(target_dir)
    redis_client = redis.Redis(host='localhost', port=6379, db=0)

    for index, (base_url, refferal_sites_res) in enumerate(tqdm(data.items())):
        if index % 100 == 0 and output_file:
            print(f"Save data at index {index} ...")
            with open(output_file, 'w') as f:
                json.dump(result, f, indent=4)

        if not refferal_sites_res:
            result[base_url] = base_url

        for ref_site in refferal_sites_res:
            if type(ref_site) == str:
                continue

            url, _ = ref_site

            if url == base_url or url in result:
                continue

            if redis_client.exists(url):
                # load from redis
                result[url] = json.loads(redis_client.get(url))
            else:
                content = alexa_scrapper.get_site_content(url)
                refferal_sites = alexa_scrapper._get_alexa_referral_sites_metric(content)
                # save in redis
                redis_client.set(url, json.dumps(refferal_sites))
                result[url] = refferal_sites

    return result

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser()
    parser.add_argument('--target_site', default='bradva.bg')

    args = parser.parse_args()

    result = ScrapeAlexa(args.target_site).scrape_alexa_site_info()

    with open(os.path.join(_DATA_PATH, f'{args.target_site}.json'), 'w') as f:
        json.dump(result, f, indent=4)

"""
Example usage:
    >>> python site_similarity/dataprep/alexa_scrapper.py --target_site "bradva.bg"
"""
