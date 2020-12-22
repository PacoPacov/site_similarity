import json
import os
from datetime import datetime

import numpy as np
import pandas as pd
from tqdm import tqdm
from utils.notebook_utils import _ALL_DATA, _DATA_PATH


def load_level_data(level=0):
    with open(f"/home/paco/Documents/site_similarity/dataset/scrapping_results/new corpus processed level {level} results.json", 'r') as f:
        data = json.load(f)

    print(f'Loaded level {level} data with key size:{len(data)} values size:{sum([len(v) for v in data.values()])}')

    return data


def normalize_sites(lvl_data, level=0):
    sites = []
    overlap_sites = []
    audience_geography = []
    levels = []
    dates = []

    for v in lvl_data.values():
        for k in v:
            sites.append(k)
            overlap_sites.append(v[k]['score'])
            audience_geography.append(v[k]['audience_geography'])
            levels.append(level)

            if os.path.exists(os.join(_ALL_DATA, f'{k}.html')):
                dates.append(
                    datetime.fromtimestamp(
                        os.stat(os.join(_ALL_DATA, f'{k}.html')).st_mtime))
            else:
                dates.append(None)
    return [sites, overlap_sites, audience_geography, levels, dates]


def load_referral_sites_from_old_data(df):
    with open(os.join(_DATA_PATH, 'clean_data_20200803.json')) as f:
        old_data = json.load(f)

    print(f"Loaded records: {len(old_data)}")

    old_df = pd.DataFrame(old_data)

    old_sites_list = old_df['sites'].tolist()
    sites = df['sites'].tolist()

    referral_sites_new_data = []
    for site in tqdm(sites):
        if site in old_sites_list:
            referral_sites_new_data.append({'sites': site,
                                            'referal_sites': old_df.loc[old_df['sites'] == site,
                                            'referal_sites'].values.tolist()[0]})
        else:
            referral_sites_new_data.append({'sites': site, 'referal_sites': None})

    referral_sites = pd.DataFrame(referral_sites_new_data)
    return df.merge(referral_sites, on='sites')


if __name__ == "__main__":
    df = pd.DataFrame([], columns=['sites', 'overlap_sites', 'audience_geography', 'levels', 'dates'])

    for index in range(5):
        lvl_data = load_level_data(level=index)
        lvl_df = (pd.DataFrame(normalize_sites(lvl_data, level=index))
                    .transpose()
                    .rename(columns={0: 'sites',
                                     1: 'overlap_sites',
                                     2: 'audience_geography',
                                     3: 'levels',
                                     4: 'dates'}))
        df = df.append(lvl_df)

    print(f'Df len {len(df)}')
    print(df.info())
    print(df.head())

    print(df.apply(lambda x: len(x['overlap_sites']), axis=1).value_counts())

    df.to_csv("/home/paco/Documents/site_similarity/dataset/new_corpus_total_data.csv", index=False)

    unique_df = df.drop_duplicates(subset=['sites'])
    print(f'Unique_df len {len(unique_df)}')

    unique_df.to_csv("/home/paco/Documents/site_similarity/dataset/new_corpus_unique_data.csv", index=False)

    res_d = load_referral_sites_from_old_data(unique_df)

    res_d['created_timestamp'] = res_d['dates'].astype(np.int64) // 10**9

    with open('clean_data_20200808.json', 'w') as f:
        json.dump(res_d[['sites', 'overlap_sites', 'referal_sites',
                         'audience_geography', 'levels', 'created_timestamp']].to_dict('records'), f, indent=4)
