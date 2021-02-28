from pathlib import Path
import yaml


def load_config(yaml_file: Path):
    with open(yaml_file) as cf_file:
        config = yaml.safe_load(cf_file.read())

    return config


def merge_dictionaries(d1, d2):
    """Update two config dictionaries recursively.
    
    :param d1: first dictionary to be updated
    :param d2: second dictionary which entries should be preferred
    """

    if d2 is None: 
        return

    for k, v in d2.items():
        if k not in d1:
            d1[k] = dict()
        if isinstance(v, dict):
            merge_dictionaries(d1[k], v)
        else:
            d1[k] = v


class ExperimentConfig:  
    """Dict wrapper that allows for slash-based retrieval of nested elements

    Example Usage:
        config.get_config("meta/dataset_name")
    """
    def __init__(self, config_path, default_path=None):
        config = load_config(config_path)
        if default_path is not None:
            default_config = load_config(default_path)
            
        merge_dictionaries(default_config, config)
        self._data = config

    def get(self, path: str=None, default=None):
        # we need to deep-copy self._data to avoid over-writing its data
        recursive_dict = dict(self._data)

        if path is None:
            return recursive_dict

        path_items = path.split("/")[:-1]
        data_item = path.split("/")[-1]

        try:
            for path_item in path_items:
                recursive_dict = recursive_dict.get(path_item)

            value = recursive_dict.get(data_item, default)

            return value
        except (TypeError, AttributeError):
            return default
