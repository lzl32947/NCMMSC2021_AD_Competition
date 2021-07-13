import os

import yaml


def create_dir(target: str) -> bool:
    """
    Create the dir and return whether the dir is created.
    :param target: str, the path to dir.
    :return: bool, whether the dir is successfully created.
    """
    if os.path.exists(target):
        return True
    else:
        try:
            # git_commit
            os.mkdir(target)
            return True
        except:
            return False


def set_working_dir(target: str):
    """
    Set working dir to target.
    :param target: str, the path to target dir.
    :return: None
    """
    os.chdir(target)


def read_config(config_path: str):
    """
    Read config.yaml to program
    :param config_path: str, the path to the config files
    :return: dict, the value of config
    """
    fin = open(config_path, encoding="utf-8")
    data = yaml.load(fin, Loader=yaml.FullLoader)
    return data


def check_dir():
    """
    Create necessary directories for program
    :return: None.
    """
    create_dir("weight")
    create_dir("log")
