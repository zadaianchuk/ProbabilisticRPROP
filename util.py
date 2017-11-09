#change the list of dict to the dict of lists
from collections import defaultdict

def get_dict_of_lists(list_of_dicts):
    dd = defaultdict(list)
    for d in list_of_dicts:
        for key, value in d.items():
            dd[key].append(value)
    return dd
