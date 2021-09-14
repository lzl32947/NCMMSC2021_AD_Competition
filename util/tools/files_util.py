import os
import shutil
import time
from typing import Dict
import warnings
import yaml

from util.log_util.logger import GlobalLogger
from util.model_util.register_util import register_model


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


def set_ignore_warning(close: bool) -> None:
    """
    Close the UserWarning by Python
    :param close: bool, whether to close the UserWarning
    :return: None
    """
    if close:
        warnings.filterwarnings("ignore")


def check_dir(configs: Dict) -> None:
    """
    Create necessary directories for program
    :param configs: Dict, the global configs
    :return: None.
    """
    # Create the 'weight' directory to store weight
    create_dir(configs["weight"]["weight_dir"])
    # Create the 'log' directory to store log
    create_dir(configs["log"]["log_dir"])
    # Create the 'image' directory to store images
    create_dir(configs["image"]["image_dir"])
    # Create the 'output' directory to store outputs
    create_dir(configs["output"]["output_dir"])


def global_init(for_evaluate: bool = False, for_competition: bool = False) -> (str, Dict):
    """
    NOTICE: THIS FUNCTION SHOULD BE RUNNING AS THE FIRST FUNCTION FOR EACH RUNNABLE.
    :param for_evaluate: bool, whether to create the directories according to evaluation, default is False
    :return: tuple(str, Dict): time string and the configs read from config.yaml
    """
    # Set the working dir to the upper
    set_working_dir("./..")
    # Generate the runtime identifier, mainly used for recording the log and weights
    run_time = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    if for_competition:
        run_time = "competition_{}".format(run_time)
    # Read the configs from the config directories, and should be in directory 'configs/config.yaml'
    config = read_config(os.path.join("configs", "config.yaml"))
    # Check and create the necessary directories
    check_dir(config)
    # Try to register all need to register
    register_model(config["model"]["model_path"])
    register_model(config["model"]["module_path"])
    # Check and create the necessary directories
    create_dir(os.path.join(config['log']['log_dir'], run_time))
    create_dir(os.path.join(config['output']['output_dir'], run_time))
    # Create the global logger and init the log with the previous config
    logger = GlobalLogger()
    logger.init_config(config['log'], run_time)
    # Check the warnings
    if 'warning' in config.keys():
        set_ignore_warning(config['warning']['ignore'])
    if not for_evaluate:
        # Create the specific weight dir
        create_dir(os.path.join(config['weight']['weight_dir'], run_time))
    else:
        create_dir(os.path.join(config['image']['image_dir'], run_time))
    # return the runtime identifier and the configs
    shutil.copy(os.path.join("configs", "config.yaml"), os.path.join(config['log']["log_dir"], run_time, "config.yaml"))
    return run_time, config
