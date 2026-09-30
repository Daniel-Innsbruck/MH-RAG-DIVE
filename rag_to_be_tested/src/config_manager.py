import yaml
import os

CONFIG_PATH = os.getenv("CONFIG_PATH", "../config.yaml")

def load_config():
    """
    Loads the YAML configuration file from the path specified by CONFIG_PATH.

    Returns:
        dict: The parsed configuration data.
    """
    with open(CONFIG_PATH, 'r', encoding='utf-8') as f:
        return yaml.load(f, Loader=yaml.Loader)

def save_config(config_data):
    """
    Saves the provided configuration data back to the YAML file.

    Args:
        config_data (dict): The configuration data to save.
    """
    with open(CONFIG_PATH, 'w', encoding='utf-8') as f:
        yaml.dump(config_data, f, allow_unicode=True)
