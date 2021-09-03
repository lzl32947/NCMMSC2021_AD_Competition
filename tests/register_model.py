import importlib
import os

from model.manager import Registers
from util.log_util.logger import GlobalLogger
from util.tools.files_util import global_init

if __name__ == '__main__':
    time_identifier, configs = global_init()
    logger = GlobalLogger().get_logger()
    for key in Registers.model.keys():
        print(key, Registers.model[key])
    for key in Registers.module.keys():
        print(key, Registers.module[key])
