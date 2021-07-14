import os
import time
import yaml

from util.log_util.logger import GlobalLogger


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


def global_init():
    """
    NOTICE: THIS FUNCTION SHOULD BE RUNNING AS THE FIRST FUNCTION FOR EACH RUNNABLE.
    :return: tuple(str, Dict): time string and the configs read from config.yaml
    """
    set_working_dir("./..")
    check_dir()
    run_time = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    config = read_config(os.path.join("configs", "config.yaml"))
    logger = GlobalLogger()
    logger.init_config(config['log'], run_time)
    return run_time, config
