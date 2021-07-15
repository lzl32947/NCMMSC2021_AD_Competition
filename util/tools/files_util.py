import os
import time
from typing import Dict

import yaml

from util.log_util.logger import GlobalLogger


def create_dir(target: str) -> bool:
    """
    Create the dir and return whether the dir is created.
    :param target: str, the path to dir.
    :return: bool, whether the dir is successfully created.
    """
    # if the directory not exist then create the directory
    if os.path.exists(target):
        return True
    else:
        try:
            os.mkdir(target)
            return True
        except:
            return False


def set_working_dir(target: str) -> None:
    """
    Set working dir to target.
    :param target: str, the path to target dir.
    :return: None
    """
    os.chdir(target)


def read_config(config_path: str) -> Dict:
    """
    Read config.yaml to program
    :param config_path: str, the path to the config files
    :return: dict, the value of config
    """
    fin = open(config_path, encoding="utf-8")
    data = yaml.load(fin, Loader=yaml.FullLoader)
    return data


def check_dir() -> None:
    """
    Create necessary directories for program
    :return: None.
    """
    # Create the 'weight' directory to store weight
    create_dir("weight")
    # Create the 'log' directory to store log
    create_dir("log")


def global_init() -> (str, Dict):
    """
    NOTICE: THIS FUNCTION SHOULD BE RUNNING AS THE FIRST FUNCTION FOR EACH RUNNABLE.
    :return: tuple(str, Dict): time string and the configs read from config.yaml
    """
    # Set the working dir to the upper
    set_working_dir("./..")
    # Check and create the necessary directories
    check_dir()
    # Generate the runtime identifier, mainly used for recording the log and weights
    run_time = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    # Read the configs from the config directories, and should be in directory 'configs/config.yaml'
    config = read_config(os.path.join("configs", "config.yaml"))
    # Create the global logger and init the log with the previous config
    logger = GlobalLogger()
    logger.init_config(config['log'], run_time)
    # return the runtime identifier and the configs
    return run_time, config
