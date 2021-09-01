import importlib
import os

from model.manager import Registers
from model.models.joint_training_model import GeneralModel
from util.log_util.logger import GlobalLogger
from util.tools.files_util import global_init

if __name__ == '__main__':
    time_identifier, configs = global_init()
    logger = GlobalLogger().get_logger()
    MODEL_MODULES = [i.replace(".py", "") for i in os.listdir("model/models")]
    ALL_MODULES = [("model.models", MODEL_MODULES), ]
    for base_dir, modules in ALL_MODULES:
        for name in modules:
            if base_dir != "":
                full_name = base_dir + "." + name
            else:
                full_name = name
            importlib.import_module(full_name)

    print(Registers.model.__dict__)
    model = Registers.model["GeneralModel"]()
    print(type(model))
    model.cuda()