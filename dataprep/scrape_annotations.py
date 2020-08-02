import csv
import os
import requests
import time
from bs4 import BeautifulSoup


MEDIA_FACTCHECKER_SECTIONS = {
    "Left": "https://mediabiasfactcheck.com/left/",
    "Left-Center": "https://mediabiasfactcheck.com/leftcenter/",
    "Least Biased": "https://mediabiasfactcheck.com/center/",
    "Right-Center": "https://mediabiasfactcheck.com/right-center/",
    "Right": "https://mediabiasfactcheck.com/right/",
    "Pro-Science": "https://mediabiasfactcheck.com/pro-science/",
    "Conspiracy-Pseudoscience": "https://mediabiasfactcheck.com/conspiracy/",
    "Questionable Sources": "https://mediabiasfactcheck.com/fake-news/",
    "Satire": "https://mediabiasfactcheck.com/satire/"
}

LABELS = ['very low', 'low', 'mixed', 'mostly factual', 'high', 'very high']

def get_mediabiasfactcheck_urls(page):
    response = requests.get(page)

    element = BeautifulSoup(response.text, 'html.parser')
    target_table = element.find('table', {'id': 'mbfc-table'})

    if not target_table:
        print("Target table couldn't be found!")
        return

    res = {a_tag.text: a_tag['href'] for a_tag in target_table.find_all('a')}
    print(f"Result size: {len(res)}")

    return res


def process_single_page(page_url):
    time.sleep(0.5)
    response = requests.get(page_url)

    parsed_page = BeautifulSoup(response.text, 'html.parser')
    article_text = parsed_page.find('article')

    result = {'original_page': page_url}

    if not article_text:
        print(f"{page_url} don't have article tag")
        return result

    paragraphs = article_text.find_all('p')

    for paragraph in paragraphs:
        if paragraph.text.startswith('Factual Reporting:'):
            spans = paragraph.find_all('span')
            for span in spans:
                if any(label in span.text.replace('\n', '').lower() for label in LABELS):
                    result['label'] = span.text.replace('\n', '')

        if any(paragraph.text.lower().strip().startswith(x) for x in ['source:', 'sources:', 'notes:', 'note:']):
            if paragraph.find('a'):
                result['url'] = paragraph.a['href']

    if not (result.get('url') and result.get('label')):
        # TODO Use the image name to find the label
        print(f'Problem with {page_url}')

    return result


if __name__ == "__main__":
    for name, url in MEDIA_FACTCHECKER_SECTIONS.items():
        if f'{name}.csv' not in os.listdir():
            print(f'Processing {name} section')
            section_urls = get_mediabiasfactcheck_urls(url)
            sections_results = [process_single_page(url) for url in section_urls.values()]

            with open(f'{name}.csv', 'w') as f:
                csv_writter = csv.DictWriter(f, fieldnames=['original_page', 'label', 'url'])
                csv_writter.writerows(sections_results)
        else:
            print(f"Skipping {name} section")
"""
Left section Result size: 317
Left-Center section Result size: 541
Least Biased section Result size: 461
Right-Center section Result size: 263
Right section Result size: 270
Pro-Science section Result size: 150
Conspiracy-Pseudoscience section Result size: 270
Questionable Sources section Result size: 487
Satire section Result size: 114
"""
