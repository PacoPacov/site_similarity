import redis
import json
from dataprep.scrape_all_alexa_information import main


def get_alexa_info(site, specific_section='site_metrics'):
    r = redis.Redis()

    target_data = None

    if r.get(site):
        target_data = json.loads(r.get(site))
    else:
        target_data = main(site)
        r.set(site, json.dumps(target_data))

    yield target_data[specific_section]
