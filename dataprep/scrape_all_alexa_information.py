import argparse
import json
import os
from collections import defaultdict

from bs4 import BeautifulSoup

from dataprep.alexa_scrapper import ScrapeAlexa


# OVERVIEW SECTION
def get_comparison_metrics(el):
    """ COMPARISON METRICS """
    card_mini_competitors = el.find('div', {'id': 'card_mini_competitors'})

    if not card_mini_competitors:
        return None


def get_similar_sites_by_audience_overlap(el):
    card_mini_audience = el.find('div', {'id': 'card_mini_audience'})

    if not card_mini_audience:
        return None

    sites = [x.text.strip(' \t\n') for x in card_mini_audience.find_all('a', {'class': 'truncation'})]
    overlap_scores = [x.text for x in card_mini_audience.find_all('span', {'class': 'truncation'})[::2]]

    return [{'site': site, 'overlap_score': overlap_score} for site, overlap_score in zip(sites, overlap_scores)]


def get_top_indusctry_topics_by_social_engagement(el):
    """ TOP INDUSTRY TOPICS BY SOCIAL ENGAGEMENT """
    top_topics = el.find('div', {"id": "card_mini_topics"})

    if not top_topics:
        return None

    topics = [topic.span.text for topic in top_topics.find_all('div', {'class': 'Showme'})]
    stat_fields = [topic.text for topic in top_topics.find_all('div', {'class': 'Third Right'})]

    average_engagement = [{"this_site": this_site,
                           "competitor_avg": competitor_avg,
                           "total_avg": total_avg}
                          for this_site, competitor_avg, total_avg in list(zip(stat_fields, stat_fields[1:], stat_fields[2:]))[::3]]
    return [{'topic': topics[index], **average_engagement[index]} for index, _ in enumerate(topics)]


def get_top_keywords_by_traffic(el):
    """ TOP KEYWORDS BY TRAFFIC """
    kws = el.find('div', {"id": "card_mini_topkw"})

    if not kws:
        return None

    keywords = [kw.span.text for kw in kws.find_all('div', {'class': 'keyword'})[1:]]
    metrics = [kw.span.text for kw in kws.find_all('div', {'class': 'metric_one'})[1:]]

    return dict(zip(keywords, metrics))


def get_alexa_rank_90_days_trends(el):
    """
    ALEXA RANK 90 DAY TREND
        * Number in global internet engagement
        * Daily Time on Site
    """
    traffic_metric = el.find('div', {'id': 'card_mini_trafficMetrics'})

    if not traffic_metric:
        return None

    return {'alexa_rank': traffic_metric.find('div', {'class': 'rankmini-global'}).div.text.strip(' \t\n'),
            'time_on_site': traffic_metric.find('div', {'class': 'rankmini-daily'}).div.text.strip(' \t\n')}

# KEYWORD OPPORTUNITIES SECTION
def get_keyword_gaps(el):
    """ Keyword Gaps """
    keyword_gaps = el.find('div', {'id': 'card_gaps'})

    if not keyword_gaps:
        return None

    keywords = [kw.span.text for kw in keyword_gaps.find_all('div', {'class': 'keyword'})[1:]]
    metric_one = [kw.span.text.strip('Avg. Traffic to Competitors') for kw in keyword_gaps.find_all('div', {'class': 'metric_one'})[1:]]
    metric_two = [kw.span.text.strip('Search Popularity') for kw in keyword_gaps.find_all('div', {'class': 'metric_two'})[1:]]

    return [{'keyword': keywords[index],
             'avg_traffic_to_competitord': metric_one[index],
             'search_popularity': metric_two[index]} for index, _ in enumerate(keywords)]


def get_easy_to_rank_keywords(el):
    """ Easy-to-Rank Keywords """
    rank_keywords = el.find('div', {'id': 'card_kwdiff'})

    if not rank_keywords:
        return None

    keywords = [kw.span.text for kw in rank_keywords.find_all('div', {'class': 'keyword'})[1:]]
    metric_one = [kw.span.text.strip('Relevance to this site') for kw in rank_keywords.find_all('div', {'class': 'metric_one'})[1:]]
    metric_two = [kw.span.text.strip('Search Popularity') for kw in rank_keywords.find_all('div', {'class': 'metric_two'})[1:]]

    return [{'keyword': keywords[index],
             'relevance_to_this_site': metric_one[index],
             'search_popularity': metric_two[index]} for index, _ in enumerate(keywords)]


def get_buyer_keywords(el):
    """ Buyer Keywords """
    card_buyer = el.find('div', {'id': 'card_buyer'})

    if not card_buyer:
        return None

    keywords = [kw.span.text for kw in card_buyer.find_all('div', {'class': 'keyword'})[1:]]
    metric_one = [kw.span.text.strip('Avg. Traffic to Competitors ') for kw in card_buyer.find_all('div', {'class': 'metric_one'})[1:]]
    metric_two = [kw.span.text.strip('Organic Competition') for kw in card_buyer.find_all('div', {'class': 'metric_two'})[1:]]

    return [{'keyword': keywords[index],
             'avg_traffic_to_competitord': metric_one[index],
             'organic_competition': metric_two[index]} for index, _ in enumerate(keywords)]


def get_optimization_opportunities(el):
    """ Optimization Opportunities """
    card_sitekw = el.find('div', {'id': 'card_sitekw'})

    if not card_sitekw:
        return None

    keywords = [kw.span.text for kw in card_sitekw.find_all('div', {'class': 'keyword'})[1:]]
    metric_one = [kw.span.text.strip('Search Popularity') for kw in card_sitekw.find_all('div', {'class': 'metric_one'})[1:]]
    metric_two = [kw.span.text.strip('Organic Share of Voice') for kw in card_sitekw.find_all('div', {'class': 'metric_two'})[1:]]

    return [{'keyword': keywords[index],
             'search_popularity': metric_one[index],
             'organic_share_of_voice': metric_two[index]} for index, _ in enumerate(keywords)]

# SOCIAL ENGAGEMENT ANALYSIS SECTION
def get_top_social_topics(el):
    """ Top Social Topicss """
    card_topics = el.find('div', {'id': 'card_topics'})

    if not card_topics:
        return None

    topics = [el for el in card_topics.find('section', {'class': 'TopicTable'}).find_all('div') if not el.get('class')]

    result = defaultdict(lambda: defaultdict(str))
    for topic in topics:
        topic_name = topic.find('h1').text
        stats = topics[0].find_all('span', {'class': 'truncation'})
        for index, stat in enumerate(stats):
            if index < 2:
                result[topic_name]['this_site' + ' ' + stat.span.text] = stat.text.replace(stat.span.text, '')
            else:
                result[topic_name]['compatitor_avg' + ' ' + stat.span.text] = stat.text.replace(stat.span.text, '')

    return result


def get_social_engagement(el):
    """ Social Engagement """
    card_socialengagement = el.find('div', {'id': 'card_socialengagement'})

    if not card_socialengagement:
        return None

    keywords = [kw.a.text for kw in card_socialengagement.find_all('div', {'class': 'site'})[1:]]
    metric_one = [kw.span.text.strip('Total Articles') for kw in card_socialengagement.find_all('div', {'class': 'metric_one'})[1:]]
    metric_two = [kw.span.text.strip('Avg. Engagement') for kw in card_socialengagement.find_all('div', {'class': 'metric_two'})[1:]]

    return [{'site': keywords[index].strip(' \t\n'),
             'total_articles': metric_one[index],
             'avg_engagement': metric_two[index]} for index, _ in enumerate(keywords)]


def get_popular_articles(el):
    """ Popular Articles """
    card_articles = el.find('div', {'id': 'card_articles'})

    if not card_articles:
        return None

    articles = [kw.a for kw in card_articles.find_all('div', {'class': 'site'})[1:]]
    metric_one = [kw.span.text.strip('Total Engagement') for kw in card_articles.find_all('div', {'class': 'metric_one'})[1:]]
    metric_two = [kw.span.text.strip('Total Shares') for kw in card_articles.find_all('div', {'class': 'metric_two'})[1:]]

    return [{'article_title': articles[index].text.strip(' \t\n'),
             'article_url': articles[index].get('href'),
             'total_articles': metric_one[index],
             'avg_engagement': metric_two[index]} for index, _ in enumerate(articles)]

# COMPETITIVE ANALYSIS SECTION
def get_traffic_sources(el):
    """ Traffic Sources """
    card_trafficsources = el.find('div', {'id': 'card_trafficsources'})

    if not card_trafficsources:
        return None

    sites = [el.text for el in card_trafficsources.find_all('div', {'class': 'Third'})[1:]]
    search_percentage = [el.span.text.strip(' \t\n') for el in card_trafficsources.find_all('div', {'class': 'ThirdFull ProgressNumberBar'})]

    return [{'site': sites[index],
             'search_percentage': search_percentage[index]} for index, _ in enumerate(sites)]


def get_referral_sites(el):
    """ Referral Sites """
    card_referralsites = el.find('div', {'id': 'card_referralsites'})

    if not card_referralsites:
        return []

    result = []

    for site_info in card_referralsites.find_all('div', {'class': 'Row'}):
        site = site_info.find('div', {'class', 'site'}).a['href'].split('/')[-1]
        number_referral_sites = site_info.find('span', {'class': 'truncation'}).text.strip()

        result.append({'site': site, 'number_referral_sites': number_referral_sites})

    return result


def get_top_keywords(el):
    """ Top Keywords """
    script = el.find('script', {'id': 'topKeywordsJSON'})

    if not script:
        return None

    # TODO find better way
    d = json.loads(vars(script)['contents'][0])

    res = {}

    for k, values in d.items():
        if k != 'titles':
            res[k] = [{'keyword': v[0]['value'],
                       'search_traffic': v[1]['value'],
                       'share_of_voice': v[2]['value']} for v in values]

    return res


def get_audience_overlap(el):
    """ Audience Overlap """
    card_overlap = el.find('div', {'id': 'card_overlap'})

    if not card_overlap:
        return None

    overlap_scores = [x.text.strip(' \t\n') for x in card_overlap.find_all('div', {'class': 'overlap'})[1:]]
    sites = [x.text.strip(' \t\n') for x in card_overlap.find_all('div', {'class': 'site'})[1:]]
    alexa_ranks = [x.text.strip(' \t\n') for x in card_overlap.find_all('div', {'class': 'metric_two'})[1:]]

    return [{'site': site, 'alexa_rank': alexa_rank, 'overlap_score': overlap_score}
            for overlap_score, site, alexa_rank in zip(overlap_scores, sites, alexa_ranks)]

# TRAFFIC STATISTICS SECTION
def get_alexa_rank(el):
    """ Alexa Rank """
    card_rank = el.find('div', {'id': 'card_rank'})

    if not card_rank:
        return None

    site_rank = card_rank.find('p', {'class': 'big data'}).text.strip(' \t\n')
    site_rank_over_past_90_days_info = card_rank.find('div', {'class': 'rank-global'}).div.span

    site_rank_over_past_90_days = site_rank_over_past_90_days_info.text

    if 'down' in site_rank_over_past_90_days_info.get('class', ''):
        site_rank_over_past_90_days = '-' + site_rank_over_past_90_days

    script_data = el.find('script', {'id': 'rankData'})
    three_month_rank_data = json.loads(vars(script_data)['contents'][0])

    country_alexa_rank_info = card_rank.find('div', {'id': 'countrydropdown'})
    country_alexa_rank = []
    for coutry_data in country_alexa_rank_info.find_all('li'):
        if coutry_data.get('data-value'):
            info = coutry_data.text.split()[1:]

            if len(info) > 2:
                country_alexa_rank.append((' '.join(info[:2]), info[-1]))
            else:
                country_alexa_rank.append((info[0], info[1]))

    return {
        'site_rank': site_rank,
        'site_rank_over_past_90_days': site_rank_over_past_90_days,
        'three_month_rank_data': three_month_rank_data.get('3mrank') if script_data else dict(),
        'coutry_alexa_ranks': country_alexa_rank
    }


def get_audience_geography_in_past_30_days(el):
    """ Audience Geography """
    card_geography = el.find('div', {'id': 'card_geography'})

    if not card_geography:
        return None

    countries = [' '.join(x.text.split('\xa0')[1:]) for x in card_geography.find_all('div', {'id': 'countryName'})]
    percents = [x.text for x in card_geography.find_all('div', {'id': 'countryPercent'})]

    return dict(zip(countries, percents))


def get_site_metrics(el):
    """ Site Metrics """
    card_metrics = el.find('div', {'id': 'card_metrics'})

    if not card_metrics:
        return None

    engagement_metrics = card_metrics.find_all('div', {'class': 'Third sectional'})

    daily_pageviews_per_visitor = engagement_metrics[0].p.text.strip(' \t\n')

    # Average time in minutes and seconds that a visitor spends on this site each day.
    daily_time_on_site, daily_time_on_site_for_the_last_90_days = engagement_metrics[1].p.text.strip(' \t\n').split(' ')
    if 'down' in engagement_metrics[1].p.span['class']:
        daily_time_on_site_for_the_last_90_days = '-' + daily_time_on_site_for_the_last_90_days

    # Percentage of visits to the site that consist of a single pageview.
    bouce_rate, bouce_rate_for_the_last_90_days = engagement_metrics[2].p.text.strip(' \t\n').split(' ')
    if 'down' in engagement_metrics[2].p.span['class']:
        bouce_rate_for_the_last_90_days = '-' + bouce_rate_for_the_last_90_days

    # The percentage of traffic that comes from both organic and paid search.
    traffic_source_search = card_metrics.find('div', {'class': "ringchart referral-social"}).get('data-referral', '')

    # Site Flow for the past 60 days
    site_flow = card_metrics.find_all('div', {'class': "Half"})

    visited_just_before = [{'site': p.text.split(' ')[1],
                            'percentage': p.span.text} for p in site_flow[0].find_all('p', {'class': 'truncation'})]

    visited_right_after = [{'site': p.text.split(' ')[1],
                            'percentage': p.span.text} for p in site_flow[1].find_all('p', {'class': 'truncation'})]

    # Sites that link to this site, recalculated weekly.
    total_sites_linking_in = card_metrics.find('span', {'class': 'big data'}).text

    return {
        'daily_pageviews_per_visitor': daily_pageviews_per_visitor,
        'daily_time_on_site': daily_time_on_site,
        'daily_time_on_site_for_the_last_90_days': daily_time_on_site_for_the_last_90_days,
        'bouce_rate': bouce_rate,
        'bouce_rate_for_the_last_90_days': bouce_rate_for_the_last_90_days,
        'traffic_source_search': traffic_source_search + '%' if traffic_source_search else traffic_source_search,
        'visited_just_before': visited_just_before,
        'visited_right_after': visited_right_after,
        'total_sites_linking_in': total_sites_linking_in
    }


def main(target_site):
    alexa = ScrapeAlexa()
    element = BeautifulSoup(alexa.get_site_content(target_site), 'lxml')

    return {
        'comparison_metrics': get_comparison_metrics(element),
        'similar_sites_by_audience_overlap': get_similar_sites_by_audience_overlap(element),
        'top_indusctry_topics_by_social_engagement': get_top_indusctry_topics_by_social_engagement(element),
        'top_keywords_by_traffic': get_top_keywords_by_traffic(element),
        'alexa_rank_90_days_trends': get_alexa_rank_90_days_trends(element),
        'keyword_gaps': get_keyword_gaps(element),
        'easy_to_rank_keywords': get_easy_to_rank_keywords(element),
        'buyer_keywords': get_buyer_keywords(element),
        'optimization_opportunities': get_optimization_opportunities(element),
        'top_social_topics': get_top_social_topics(element),
        'social_engagement': get_social_engagement(element),
        'popular_articles': get_popular_articles(element),
        'traffic_sources': get_traffic_sources(element),
        'referral_sites': get_referral_sites(element),
        'top_keywords': get_top_keywords(element),
        'audience_overlap': get_audience_overlap(element),
        'alexa_rank': get_alexa_rank(element),
        'audience_geography_in_past_30_days': get_audience_geography_in_past_30_days(element),
        'site_metrics': get_site_metrics(element)
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Extracts the whole infromation from Alexa for target site and saves it in JSON file.')

    parser.add_argument('target_site', type=str, help='Name of the site. Example: cnn.com')
    parser.add_argument('output_dir', type=str, help='Where to save the extracted information')

    args = parser.parse_args()

    extracted_information = main(args.target_site)

    os.makedirs(args.output_dir, exist_ok=True)
    output_file_path = os.path.join(args.output_dir, f'{args.arget_site}.json')

    with open(output_file_path, 'w') as f:
        json.dump(extracted_information, f)


"""
Before running the code you should add the path to the project in your PYTHONPATH

On Windows is like:
    set PYTHONPATH=%PYTHONPATH%;C:\\PATH_TO_site_similarity

On Linux is like:
    export PYTHONPATH=\home\PATH_TO_site_similarity:$PYTHONPATH
"""
