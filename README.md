# Site Similarity project
The main goal in this project is to have a tool that can give you more information about particular site.

We use the scrapped data from this [repo](https://github.com/ramybaly/News-Media-Reliability).

### Setting it up
1. Clone the repo
    ```
    >>> git clone https://github.com/PacoPacov/site_similarity.git
    ```
2. Create virtual env
    ```
    >>> pytho36 -m venv venv
    >>> source venv/bin/activate
    >>> pip install -r site_similarity/requirements.txt
    ```
3. Create folder 'data' in path/to/project/site_similarity/site_similarity
4. In the data folder please create another folder - 'annotated_data' and download the three files from this [repo](https://github.com/ramybaly/News-Media-Reliability/tree/master/data) there.
    * corpus.csv
    * splits.json
    * stats.txt
    * Note that the scripts were ran on the data from (24 August 2018)
