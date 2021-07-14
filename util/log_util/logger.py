import logging
import threading
from typing import Dict
import os
import time


class GlobalLogger(object):
    _instance_lock = threading.Lock()

    def __init__(self):
        if not hasattr(self, "log"):
            self.log = None
        if not hasattr(self, "config"):
            self.config = None
        if not hasattr(self, "log_name"):
            self.log_name = None

    def __new__(cls, *args, **kwargs):
        if not hasattr(GlobalLogger, "_instance"):
            with GlobalLogger._instance_lock:
                if not hasattr(GlobalLogger, "_instance"):
                    GlobalLogger._instance = object.__new__(cls)
        return GlobalLogger._instance

    def config_logger(self):
        self.log.setLevel(logging.INFO)

    def get_logger(self) -> logging.Logger:
        return self.log

    def get_config(self) -> Dict:
        return self.config

    def init_config(self, config, store_name):
        self.log = logging.getLogger("main")
        self.log.setLevel(logging.DEBUG)
        self.config = config
        self.log_name = store_name

        ch = logging.StreamHandler()
        ch.setLevel(logging.DEBUG)
        formatter = logging.Formatter("%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s")
        ch.setFormatter(formatter)
        self.log.addHandler(ch)

        if not os.path.exists(os.path.join(self.config["log_dir"], self.log_name)):
            os.mkdir(os.path.join(self.config["log_dir"], self.log_name))
        fh = logging.FileHandler(os.path.join(self.config['log_dir'], self.log_name, "run.log"), mode="w")
        fh.setLevel(logging.INFO)
        formatter = logging.Formatter("%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s")
        fh.setFormatter(formatter)
        self.log.addHandler(fh)
