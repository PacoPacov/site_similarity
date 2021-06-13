import json
import os

import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
import redis
from bs4 import BeautifulSoup
from stellargraph import StellarGraph

from dataprep.alexa_scrapper import ScrapeAlexa
from dataprep.scrape_all_alexa_information import main
from dataprep.load_annotated_data import load_corpus

_PROJECT_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_DATA_PATH = os.path.join(_PROJECT_PATH, 'data')
_ALL_DATA = os.path.join(_PROJECT_PATH, 'dataset', 'all_data')
_FEATURES_DIR = os.path.join(_DATA_PATH, 'features')
_ALEXA_SECTIONS_NAMES = {
    'comparison_metrics': None,
    'similar_sites_by_audience_overlap': None,
    'top_industry_topics_by_social_engagement': None,
    'top_keywords_by_traffic': None,
    'alexa_rank_90_days_trends': ['alexa_rank', 'time_on_site'],
    'keyword_gaps': None,
    'easy_to_rank_keywords': None,
    'buyer_keywords': None,
    'optimization_opportunities': None,
    'top_social_topics': None,
    'social_engagement': None,
    'popular_articles': None,
    'traffic_sources': None,
    'referral_sites': None,
    'top_keywords': None,
    'audience_overlap': None,
    'alexa_rank': ['site_rank', 'site_rank_over_past_90_days',
                   'three_month_rank_data', 'country_alexa_ranks'],
    'audience_geography_in_past_30_days': None,
    'site_metrics': [
        'daily_pageviews_per_visitor', 'daily_pageviews_per_visitor_for_the_last_90_days',
        'daily_time_on_site', 'daily_time_on_site_for_the_last_90_days', 'bounce_rate',
        'bounce_rate_for_the_last_90_days', 'traffic_source_search', 'visited_just_before',
        'visited_right_after', 'total_sites_linking_in']
}


def load_level_data(data_path=None, level=0):
    if not data_path:
        data_path = os.path.join(_DATA_PATH, 'clean_data_20200803.json')

    with open(data_path, 'r') as f:
        data = json.load(f)

    output = {record['sites']: record for record in data if record['levels'] <= level}
    print((f"Loaded {len(output)} nodes with records level <= {level} and child size:"
           f"{sum([len(record['overlap_sites']) for record in output.values()])}"))

    return output


def create_nodes(lvl_data, edge_type=None):
    nodes = []
    for k in list(lvl_data.keys()):
        if not lvl_data[k]['overlap_sites']:
            el = (k, k, edge_type) if edge_type else (k, k)
            nodes.append(el)
        else:
            for urls in lvl_data[k]['overlap_sites']:
                el = (k, urls['url'], edge_type) if edge_type else (k, urls['url'])
                nodes.append(el)
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


def combined_nodes_referral_sites_audience_overlap(data_year='2020', level=1, add_edge_type=False):
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
            el = (base_url, base_url, 'referral_site_to') if add_edge_type else (base_url, base_url)
            referral_sites_NODES.append(el)

        for referral_site, _ in referral_sites:
            if referral_site != base_url:
                el = (base_url, referral_site, 'referral_site_to') if add_edge_type else (base_url, referral_site)
                referral_sites_NODES.append(el)

    audience_overlap_sites = load_level_data(os.path.join(_DATA_PATH, audience_overlap_scrapping_file), level=level)

    if add_edge_type:
        audience_overlap_sites_NODES = create_nodes(audience_overlap_sites, edge_type='similar_by_audience_overlap_to')
    else:
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


def check_sections_population(result):
    population_info = {}
    for section, fields in _ALEXA_SECTIONS_NAMES.items():
        if result.get(section):
            population_info[section] = 1
            if fields:
                for field in fields:
                    population_info[f'{section}_{field}'] = int(bool(result.get(section).get(field)))
        else:
            population_info[section] = 0

    return population_info


def extact_data_needed_for_for_node_feature_report():
    r = redis.Redis()

    all_keys = r.keys()

    report_data = {}
    for key in all_keys:
        normalized_key = key.decode('utf-8')
        report_data[normalized_key] = check_sections_population(json.loads(r.get(normalized_key)))

    df = pd.DataFrame(report_data)

    # save the data
    df.T.to_csv('report_all_sites_alexa_section_population.csv')


def generate_alexa_metric_indicator(data_year):
    r = redis.Redis()

    has_daily_pageviews_per_visitor = {}
    print('What', type(has_daily_pageviews_per_visitor))
    has_daily_time_on_site = {}
    has_bounce_rate = {}
    has_total_sites_linking_in = {}
    has_alexa_rank = {}

    if data_year == '2020':
        corpus = [site['source_url_processed'].strip() for site in load_corpus('new_corpus_2020.csv', data_year='2020')]
    elif data_year == '2018':
        corpus = [site['source_url_processed'].strip() for site in load_corpus('corpus_2018_20200907.tsv', data_year='2018', delimiter='\t')]
    else:
        raise ValueError(f'Incorrect parameter "data_year" = {data_year}')

    for site in corpus:
        print(site)
        result = json.loads(r.get(site))

        if not result.get('site_metrics'):
            print('Missing site_metrics for:', site)
            print('What', type(has_daily_pageviews_per_visitor))
            has_daily_pageviews_per_visitor[site] = int(False)
            has_daily_time_on_site[site] = int(False)
            has_bounce_rate[site] = int(False)
            has_total_sites_linking_in[site] = int(False)

            continue

        has_daily_pageviews_per_visitor[site] = [int(bool(result['site_metrics'].get('daily_pageviews_per_visitor')))]
        has_daily_time_on_site[site] = [int(bool(result['site_metrics'].get('daily_time_on_site')))]
        has_bounce_rate[site] = [int(bool(result['site_metrics'].get('bounce_rate')))]
        has_total_sites_linking_in[site] = [int(bool(result['site_metrics'].get('total_sites_linking_in')))]

    print('Finished with site_metrics')

    for site in corpus:
        result = json.loads(r.get(site))

        if not result.get('alexa_rank'):
            has_alexa_rank[site] = int(False)
            continue

        has_alexa_rank[site] = [int(bool(result['alexa_rank'].get('site_rank')))]


    print('Finished with Alexa rank')

    if data_year == '2018':
        correct_mapping = {
                "conservativeoutfitters.com": "conservativeoutfitters.com-blogs-news",
                "who.int": "who.int-en",
                "hemaven.net": "themaven.net-beingliberal",
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
        for k, v in correct_mapping.items():
            has_daily_pageviews_per_visitor[v] = has_daily_pageviews_per_visitor.pop(k)
            has_daily_time_on_site[v] = has_daily_time_on_site.pop(k)
            has_bounce_rate[v] = has_bounce_rate.pop(k)
            has_total_sites_linking_in[v] = has_total_sites_linking_in.pop(k)
            has_alexa_rank[v] = has_alexa_rank.pop(k)

    files = {file.format(data_year): feature
             for file, feature in {'has_daily_pageviews_per_visitor_{}.json': has_daily_pageviews_per_visitor,
                             'has_daily_time_on_site_{}.json': has_daily_time_on_site,
                             'has_bounce_rate_{}.json': has_bounce_rate,
                             'has_total_sites_linking_in_{}.json': has_total_sites_linking_in,
                             'has_alexa_rank_{}.json': has_alexa_rank}.items()}

    for file_name, feature in files.items():
        with open(os.path.join(_FEATURES_DIR, file_name), 'w') as f:
            json.dump(feature, f)