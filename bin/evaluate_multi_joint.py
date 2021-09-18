import os
import time
from typing import Dict, Union, List
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as func
from sklearn.metrics import confusion_matrix

from bin.evaluate_joint import evaluate_joint
from bin.evaluate_spec import evaluate_specific
from configs.types import AudioFeatures, DatasetMode, ADType
from model.manager import Registers
from util.log_util.logger import GlobalLogger
from util.tools.files_util import global_init
from util.train_util.trainer_util import prepare_dataloader, get_best_loss_weight
from tqdm import tqdm
import numpy as np
import pickle

if __name__ == '__main__':
    time_identifier, configs = global_init(for_evaluate=True, for_competition=True)
    logger = GlobalLogger().get_logger()
    feature_list = [AudioFeatures.SPECS, AudioFeatures.MFCC, AudioFeatures.MELSPECS]
    model_list = [
        {"model_name": "CompetitionMSMJointConcatFineTuneLongModel",
         "weight_identifier": "competition_20210918_145253",
         "input_channels": 3,
         "use_features": feature_list,
         "weight_description": "General",
         "joint": True},
        {"model_name": "CompetitionMSMJointConcatFineTuneLongModel",
         "weight_identifier": "competition_20210918_145253",
         "input_channels": 3,
         "use_features": feature_list,
         "weight_description": "Fine_tune",
         "joint": True},
        {"model_name": "CompetitionSpecificTrainModel",
         "weight_identifier": "competition_20210918_145253",
         "input_channels": 1,
         "use_features": AudioFeatures.MELSPECS,
         "weight_description": None,
         "joint": False},
        {"model_name": "CompetitionSpecificTrainModel",
         "weight_identifier": "competition_20210918_145253",
         "input_channels": 1,
         "use_features": AudioFeatures.SPECS,
         "weight_description": None,
         "joint": False},
        {"model_name": "CompetitionSpecificTrainModel",
         "weight_identifier": "competition_20210918_145253",
         "input_channels": 1,
         "use_features": AudioFeatures.MFCC,
         "weight_description": None,
         "joint": False},

        {"model_name": "CompetitionSpecificTrainVggNet16BNBackboneModel",
         "weight_identifier": "competition_20210918_104334",
         "input_channels": 3,
         "use_features": AudioFeatures.SPECS,
         "weight_description": None,
         "joint":False },
        {"model_name": "CompetitionSpecificTrainVggNet16BNBackboneModel",
         "weight_identifier": "competition_20210918_104334",
         "input_channels": 3,
         "use_features": AudioFeatures.MFCC,
         "weight_description": None,
         "joint": False},
        {"model_name": "CompetitionSpecificTrainVggNet16BNBackboneModel",
         "weight_identifier": "competition_20210918_104334",
         "input_channels": 3,
         "use_features": AudioFeatures.MELSPECS,
         "weight_description": None,
         "joint": False},
        {"model_name": "CompetitionSpecificTrainVggNet19BNBackboneModel",
         "weight_identifier": "competition_20210918_114007",
         "input_channels": 3,
         "use_features": AudioFeatures.SPECS,
         "weight_description": None,
         "joint": False},
        {"model_name": "CompetitionSpecificTrainVggNet19BNBackboneModel",
         "weight_identifier": "competition_20210918_114007",
         "input_channels": 3,
         "use_features": AudioFeatures.MFCC,
         "weight_description": None,
         "joint": False},
        {"model_name": "CompetitionSpecificTrainVggNet19BNBackboneModel",
         "weight_identifier": "competition_20210918_114007",
         "input_channels": 3,
         "use_features": AudioFeatures.MELSPECS,
         "weight_description": None,
         "joint": False},

    ]
    result_only = True
    for_post = True
    dataset_mode = DatasetMode.EVAL5
    output_list = []

    for model in model_list:
        logger.info("Using {} for evaluation".format(model["model_name"]))
        if model["joint"]:
            d = evaluate_joint(time_identifier, configs, model["model_name"],
                               model["use_features"],
                               model["weight_identifier"], model["weight_description"], dataset_mode)
        else:
            d = evaluate_specific(time_identifier, configs, model["model_name"], model["use_features"]
                                  , model["weight_identifier"], dataset_mode, input_channels=model["input_channels"])
        output_list.append(d)
        # print(d)

    output = os.path.join(configs["output"]["output_dir"], time_identifier,
                          "{}_{}.txt".format("Multi-model", dataset_mode.value))
    mapping_dict = {0: "AD", 1: "HC", 2: "MCI"}
    standard_dict = {"AD": 1, "MCI": 2, "HC": 3}
    keys = list(output_list[0].keys())
    list.sort(keys)
    with open(output, "w", encoding="utf-8") as fout:
        if for_post:
            fout.write("ID Prediction\n")
        for item in keys:
            file_name = item.split("/")[-1] if "/" in item else item.split("\\")[-1]
            array_average = np.average([o[item] for o in output_list], axis=0)
            prediction = int(np.argmax(array_average, axis=0))
            if result_only and not for_post:
                write_line = "{} {}\n".format(file_name, mapping_dict[prediction])
            elif not for_post:
                write_line = "{} {} {}\n".format(file_name, mapping_dict[prediction],
                                                 float(np.average([o[item] for o in output_list], axis=0)[prediction]))
            else:
                write_line = "{} {}\n".format(file_name, standard_dict[mapping_dict[prediction]])
            fout.write(write_line)
    logger.info("Saving results to {}".format(output))
