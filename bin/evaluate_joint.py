import os
import time
from typing import Dict, Union, List
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as func
from sklearn.metrics import confusion_matrix

from configs.types import AudioFeatures, DatasetMode, ADType
from model.manager import Registers
from util.log_util.logger import GlobalLogger
from util.tools.files_util import global_init
from util.train_util.trainer_util import prepare_dataloader, get_best_loss_weight
from tqdm import tqdm
import numpy as np
import pickle


def evaluate_joint(identifier: str, config: Dict, model_name: str, use_feature: List[AudioFeatures],
                   weight_identifier: str, weight_description: str, run_for: DatasetMode, input_channels: int = 1,
                   **kwargs):
    predicted_label = []
    predicted_file = []

    # Get the fold
    logger = GlobalLogger().get_logger()
    logger.info("Evaluate the model {}".format(model_name))
    for test_dataloader in prepare_dataloader(use_feature, config["dataset"], run_for, k_fold=0,
                                              repeat_times=5):
        # Get model and send to GPU
        model = Registers.model[model_name]()
        model = model.cuda()
        # Load weight file into model
        weight_file = get_best_loss_weight(
            os.path.join(config['weight']['weight_dir'], weight_identifier), weight_description)
        logger.info("Using weight {}".format(weight_file))
        model.load_state_dict(torch.load(weight_file), strict=True)
        # Set to eval mode
        model.eval()

        # Get the length of the test dataloader
        length = len(test_dataloader)

        # Init the bar
        bar_test = tqdm(range(length))
        bar_test.set_description(
            "Testing {}".format(weight_description))


        # Re-init the timer
        current_time = time.time()

        with torch.no_grad():
            # Running one batch
            for data in test_dataloader:
                # Get the features
                feature_list = []
                name = data[AudioFeatures.NAME]
                for item in use_feature:

                    feature = data[item]
                    if input_channels != 1:
                        feature = torch.cat([feature] * input_channels, dim=1)
                    feature = feature.cuda()
                    feature_list.append(feature)

                predicted_file.append(name)

                # Running the model
                output = model(*feature_list)
                # Normalize the output to one-hot mode
                predicted = func.softmax(output, dim=1)

                predicted_label.append(predicted.cpu().numpy())

                # Update the bar
                bar_test.update(1)

        bar_test.close()
        # Time the timer
        now_time = time.time()
        logger.info(
            "Finish testing {}, time cost {:.2f}s".format(
                weight_description,
                now_time - current_time))
    predicted_label = np.concatenate(predicted_label, axis=0)
    predicted_file = np.concatenate(predicted_file, axis=0)

    assert len(predicted_label) == len(predicted_file)
    output_dict = {}
    for item in set(predicted_file):
        prediction_output = predicted_label[predicted_file == item]
        average_output = np.average(prediction_output, axis=0)
        output_dict[str(item)] = average_output

    return output_dict


if __name__ == '__main__':
    time_identifier, configs = global_init(for_evaluate=True, for_competition=True)
    logger = GlobalLogger().get_logger()
    model_name = "CompetitionMSMJointConcatFineTuneLongModel"
    weight_identifier = "competition_20210914_172301"
    d = evaluate_joint(time_identifier, configs, model_name,
                       [AudioFeatures.MFCC, AudioFeatures.SPECS, AudioFeatures.MELSPECS],
                       weight_identifier, "General", DatasetMode.EVAL30)
    print(d)
    logger.info("Analysis results for {} with {}".format(model_name, weight_identifier))
