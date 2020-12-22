import json

import pandas as pd

from dataprep.scrape_all_alexa_information import main


file_name = "data_for_trainig_model_corpus_2018_audience_overlap_sites_level_3_and_referral_data_2018_corpus_level3_deep.csv"

df = pd.read_csv(file_name)
df.head()

unique_sources = df.source.unique().tolist()
unique_targets = df.target.unique().tolist()

sources = set(unique_sources).union(set(unique_targets))

for index, site_name in enumerate(sources):
    if index and index % 100 == 0:
        print(f'Processed: {index} elements')
    try:
        res = {site_name: main(site_name)}
    except Exception as e:
        print(f'Problem with site: {site_name}')
        print(e)
        continue

    with open('extracted_alexarank_data.json', 'a') as f:
        json.dump(res, f)
        f.write('\n')

# save sources in a file for easier check in the feature
with open('extracted_alexarank_data_ids.json', 'w') as f:
    json.dump(list(sources), f)
