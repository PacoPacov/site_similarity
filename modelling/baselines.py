from dataprep.load_annotated_data import apply_splits, load_corpus


def baseline_one(data):
    """
    When we have equal or no information about relative sites: We take the most frequent one from the data
    :param data:
    :return:
    """
    # TODO implement the baseline
    pass


def baseline_two(data):
    """
    We look at overlap score and take the label of this site
    :param data:
    :return:
    """
    # TODO implement the baseline
    pass


def baseline_three(data):
    """
    We don't look at votes we take the label from the first annotated related site.
    :param data:
    :return:
    """
    # TODO implement the baseline
    pass


if __name__ == '__main__':
    DATA = load_corpus()
    SPLITS = apply_splits(DATA)

    print(SPLITS.keys())
