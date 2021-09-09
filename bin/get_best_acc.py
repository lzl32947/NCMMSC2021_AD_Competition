from configs.types import AudioFeatures
from util.log_util.logger import GlobalLogger
from util.tools.files_util import global_init
from util.train_util.trainer_util import get_best_acc_weight
import os
import re


def get_acc_only(config, target_directory, feature_name):
    fold = config["dataset"]["k_fold"]

    for i in range(fold):
        weight_file = get_best_acc_weight(
            os.path.join(configs['weight']['weight_dir'], target_directory),
            fold, i, feature_name)
        print(weight_file.replace("\\", "/"))


def get_acc_for_log(config, target_directory, feature_name):
    fold = config["dataset"]["k_fold"]

    format_str = ""
    for i in range(fold):
        weight_file = get_best_acc_weight(
            os.path.join(configs['weight']['weight_dir'], target_directory),
            fold, i, feature_name)
        weight_file_name = weight_file.replace("\\", "/").split("/")[-1]
        acc = weight_file_name.split("-")[-1][3:-4]
        acc = float(acc)
        s = "[{:.2f}%]({})".format(acc * 100, weight_file.replace("\\", "/"))
        format_str += s
        if i != fold - 1:
            format_str += ","
    print(format_str)


if __name__ == '__main__':
    time_identifier, configs = global_init()
    logger = GlobalLogger().get_logger()

    target_directory = "20210907_230704"
    feature_name = AudioFeatures.MELSPECS_VAD.value
    get_acc_for_log(configs, target_directory, feature_name)
