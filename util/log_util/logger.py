import logging
import threading
from typing import Dict
import os
import time


class GlobalLogger(object):
    """
    This is the class of the global logger and is set to be the single-instance when running
    """
    # Add lock in the instance for threading in case
    _instance_lock = threading.Lock()

    def __init__(self) -> None:
        # Check the attr in __init__ for not initializing the parameters in re-creating the instance
        if not hasattr(self, "log"):
            self.log = None
        if not hasattr(self, "config"):
            self.config = None
        if not hasattr(self, "log_name"):
            self.log_name = None

    def __new__(cls, *args, **kwargs):
        # This function is used to lock the instance for only create the single instance when running
        if not hasattr(GlobalLogger, "_instance"):
            with GlobalLogger._instance_lock:
                if not hasattr(GlobalLogger, "_instance"):
                    GlobalLogger._instance = object.__new__(cls)
        return GlobalLogger._instance

    def get_logger(self) -> logging.Logger:
        """
        Get the global logger.
        :return: logging.Logger instance, and in class return the instance of GlobalLogger
        """
        return self.log

    def get_config(self) -> Dict:
        """
        Return the configuration of logger.
        :return: Dict, config files of the logger
        """
        return self.config

    def init_config(self, config: Dict, store_name: str) -> None:
        """
        Init the logger with the given parameters.
        :param config: Dict, the configs
        :param store_name: str, the name of the saving name
        :return: None
        """
        # Set logger name
        self.log = logging.getLogger("main")
        # Set logger level
        self.log.setLevel(logging.DEBUG)
        self.config = config
        self.log_name = store_name
        # Init the console logger
        ch = logging.StreamHandler()
        ch.setLevel(logging.DEBUG)
        formatter = logging.Formatter("%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s")
        ch.setFormatter(formatter)
        self.log.addHandler(ch)
        # Create the log directory with the store_name if not exist(and should be not exist)
        if not os.path.exists(os.path.join(self.config["log_dir"], self.log_name)):
            os.mkdir(os.path.join(self.config["log_dir"], self.log_name))
        # Init the file logger
        fh = logging.FileHandler(os.path.join(self.config['log_dir'], self.log_name, "run.log"), mode="w")
        fh.setLevel(logging.INFO)
        formatter = logging.Formatter("%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s")
        fh.setFormatter(formatter)
        self.log.addHandler(fh)
