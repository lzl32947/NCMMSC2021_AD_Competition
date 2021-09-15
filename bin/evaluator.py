import os
import time
from typing import Dict, Union, List, Callable
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as func
from sklearn.metrics import confusion_matrix

from configs.types import AudioFeatures, DatasetMode, ADType
from model.manager import Registers
from util.log_util.logger import GlobalLogger
from util.tools.files_util import global_init
from util.train_util.data_loader import AldsTorchDataset
from util.train_util.trainer_util import prepare_dataloader, get_best_acc_weight
from tqdm import tqdm
import numpy as np
import pickle


def evaluate_specific(identifier: str, config: Dict, model_name: str, use_feature: Union[AudioFeatures, str],
                      weight_identifier: str,dataset_func:Callable, input_channels: int = 1, **kwargs):
    correct_label = []
    predicted_label = []
    total = 0
    correct = 0

    # Get the fold
    total_fold = config['dataset']['k_fold']
    logger = GlobalLogger().get_logger()
    logger.info("Evaluate the model {}.".format(model_name))
    for current_fold, test_dataloader in enumerate(
            prepare_dataloader([use_feature], config["dataset"], DatasetMode.TEST,
                               repeat_times=5 * config["dataset"]["repeat_times"], dataset_func=dataset_func)):
        # Get model and send to GPU
        model = Registers.model[model_name](**kwargs)
        model = model.cuda()
        # Load weight file into model
        weight_file = get_best_acc_weight(
            os.path.join(config['weight']['weight_dir'], weight_identifier),
            total_fold, current_fold, use_feature.value if isinstance(use_feature, AudioFeatures) else use_feature)
        logger.info("Using weight {}".format(weight_file))
        model.load_state_dict(torch.load(weight_file), strict=True)
        # Set to eval mode
        model.eval()

        # Get the length of the test dataloader
        length = len(test_dataloader)

        # Init the bar
        bar_test = tqdm(range(length))
        bar_test.set_description(
            "Testing using feature {}, for fold {}/{}.".format(use_feature, current_fold,
                                                               total_fold))

        correct_label_fold = []
        predicted_label_fold = []

        # Re-init the timer
        current_time = time.time()

        with torch.no_grad():
            # Running one batch
            for data in test_dataloader:
                # Get the features

                feature, label = data[use_feature], data[AudioFeatures.LABEL]

                if input_channels != 1:
                    feature = torch.cat([feature] * input_channels, dim=1)

                correct_label_fold.append(label.numpy())

                feature = feature.cuda()
                label = label.cuda()
                # Running the model
                output = model(feature)
                # Normalize the output to one-hot mode
                _, predicted = torch.max(func.softmax(output, dim=1), 1)

                predicted_label_fold.append(predicted.cpu().numpy())

                # Record the size
                total += label.size(0)
                # Record the correct output
                correct += (predicted == label).sum().item()
                # Calculate the accuracy
                acc = correct / total
                # Visualize the accuracy
                bar_test.set_postfix(acc=acc)
                # Update the bar
                bar_test.update(1)

            correct_label.append(correct_label_fold)
            predicted_label.append(predicted_label_fold)
        bar_test.close()
        # Time the timer
        now_time = time.time()
        logger.info(
            "Finish testing feature {}, for fold {}/{}, time cost {:.2f}s ,with acc {:.2f}%".format(
                use_feature.value if isinstance(use_feature, AudioFeatures) else use_feature,
                current_fold,
                total_fold,
                now_time - current_time,
                acc * 100))
    return correct_label, predicted_label


def evaluate_joint(identifier: str, config: Dict, model_name: str, use_feature: List[AudioFeatures],
                   weight_identifier: str, weight_description: str, dataset_func:Callable,input_channels: int = 1, **kwargs):
    correct_label = []
    predicted_label = []
    total = 0
    correct = 0

    # Get the fold
    total_fold = config['dataset']['k_fold']
    logger = GlobalLogger().get_logger()
    logger.info("Evaluate the model {}.".format(model_name))
    for current_fold, test_dataloader in enumerate(
            prepare_dataloader(use_feature, config["dataset"], DatasetMode.TEST,dataset_func=dataset_func,
                               repeat_times=5 * config["dataset"]["repeat_times"])):
        # Get model and send to GPU
        model = Registers.model[model_name](**kwargs)
        model = model.cuda()
        # Load weight file into model
        weight_file = get_best_acc_weight(
            os.path.join(config['weight']['weight_dir'], weight_identifier),
            total_fold, current_fold, weight_description)
        logger.info("Using weight {}".format(weight_file))
        model.load_state_dict(torch.load(weight_file), strict=True)
        # Set to eval mode
        model.eval()

        # Get the length of the test dataloader
        length = len(test_dataloader)

        # Init the bar
        bar_test = tqdm(range(length))
        bar_test.set_description(
            "Testing {}, for fold {}/{}.".format(weight_description, current_fold,
                                                 total_fold))

        correct_label_fold = []
        predicted_label_fold = []

        # Re-init the timer
        current_time = time.time()

        with torch.no_grad():
            # Running one batch
            for data in test_dataloader:
                # Get the features
                feature_list = []
                label = data[AudioFeatures.LABEL]
                for item in use_feature:

                    feature = data[item]
                    if input_channels != 1:
                        feature = torch.cat([feature] * input_channels, dim=1)
                    feature = feature.cuda()
                    feature_list.append(feature)
                correct_label_fold.append(label.numpy())

                label = label.cuda()
                # Running the model
                output = model(*feature_list)
                # Normalize the output to one-hot mode
                _, predicted = torch.max(func.softmax(output, dim=1), 1)

                predicted_label_fold.append(predicted.cpu().numpy())

                # Record the size
                total += label.size(0)
                # Record the correct output
                correct += (predicted == label).sum().item()
                # Calculate the accuracy
                acc = correct / total
                # Visualize the accuracy
                bar_test.set_postfix(acc=acc)
                # Update the bar
                bar_test.update(1)

            correct_label.append(correct_label_fold)
            predicted_label.append(predicted_label_fold)
        bar_test.close()
        # Time the timer
        now_time = time.time()
        logger.info(
            "Finish testing {}, for fold {}/{}, time cost {:.2f}s ,with acc {:.2f}%".format(
                weight_description,
                current_fold,
                total_fold,
                now_time - current_time,
                acc * 100))
    return correct_label, predicted_label


def plot_image(identifier, config, cm, classes, title: str, cmap=plt.cm.Blues):
    # Set font and size
    plt.rc('font', family='Times New Roman', size='16')

    # Normalized
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    # Remove the percent lower than 1%
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            if int(cm[i, j] * 100 + 0.5) == 0:
                cm[i, j] = 0

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    # Draw the color bar
    # ax.figure.colorbar(im, ax=ax)

    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='Actual',
           xlabel='Predicted')

    # Draw the grids
    ax.set_xticks(np.arange(cm.shape[1] + 1) - .5, minor=True)
    ax.set_yticks(np.arange(cm.shape[0] + 1) - .5, minor=True)
    ax.grid(which="minor", color="gray", linestyle='-', linewidth=0.2)
    ax.tick_params(which="minor", bottom=False, left=False)

    # Rotate the X label for 45 degrees
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Percentage information
    fmt = 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            if int(cm[i, j] * 100 + 0.5) > 0:
                ax.text(j, i, format(int(cm[i, j] * 100 + 0.5), fmt) + '%',
                        ha="center", va="center",
                        color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    fig.savefig(os.path.join(config["image"]["image_dir"], identifier,
                             "{}.png".format(title.replace(" ", "_").replace("\n", "_"))),
                transparent=True)
    plt.close(fig)


def analysis_result(identifier, config, correct_label, predicted_label, model_name):
    assert len(correct_label) == len(predicted_label)
    total_fold = len(correct_label)
    total_correct = []
    total_predicted = []
    for fold in range(total_fold):
        correct = np.concatenate(correct_label[fold])
        total_correct.append(correct)
        predicted = np.concatenate(predicted_label[fold])
        total_predicted.append(predicted)
        acc = len(correct[correct == predicted]) / len(correct)
        matrix = confusion_matrix(y_true=correct, y_pred=predicted)
        plot_image(identifier, config, matrix, [i.value for i in ADType],
                   "{} {}-{} Fold Results\nAccuracy {:.2f} Percent".format(model_name, fold, total_fold, acc * 100))
        logger.info("Finish generating image {}-{}.".format(fold, total_fold))
    correct = np.concatenate(total_correct)
    predicted = np.concatenate(total_predicted)
    matrix = confusion_matrix(y_true=correct, y_pred=predicted)
    acc = len(correct[correct == predicted]) / len(correct)
    plot_image(identifier, config, matrix, [i.value for i in ADType],
               "{} Results\nAccuracy {:.2f} Percent".format(model_name, acc * 100))
    logger.info("Finish generating all images.")


if __name__ == '__main__':
    time_identifier, configs = global_init(True)
    logger = GlobalLogger().get_logger()
    model_name = "CompetitionSpecificTrainVggNet19BNBackboneModel"
    weight_identifier = "20210915_093218"
    # c, p = evaluate_joint(time_identifier, configs, model_name,
    #                       [AudioFeatures.MFCC, AudioFeatures.SPECS, AudioFeatures.MELSPECS], weight_identifier,
    #                       "Fine_tune", input_shape=())
    c, p = evaluate_specific(time_identifier, configs, model_name,
                             AudioFeatures.SPECS, weight_identifier,AldsTorchDataset,input_channels=3)
    logger.info("Analysis results for {} with {}".format(model_name, weight_identifier))
    analysis_result(time_identifier, configs, c, p, model_name)
