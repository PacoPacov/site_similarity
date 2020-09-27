import os

import redis
from tqdm import tqdm
from dataprep.alexa_scrapper import _DATA_PATH


r = redis.Redis()

for response_file in tqdm(os.listdir(_DATA_PATH)):

    with open(os.path.join(_DATA_PATH, response_file), 'r') as f:
        data = f.read()

    if not r.exists(response_file):
        r.set(response_file, data)
    else:
        print(f'The key:{response_file} is already in Redis.')


"""
The main idea of this script is to insert all saved html results in Redis.
That way we will reduce the time needed for generating new features because there won't be any reading of a file.
NOTE: Look for your RAM because Redis is storing everything in it.
To run:
>>> export PYTHONPATH=/PATH_TO_PROJECT/site_similarity
>>> python insert_all_responses_to_redis.py
"""
