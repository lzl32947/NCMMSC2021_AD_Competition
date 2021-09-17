import os
import time
from typing import Dict, Union, List, Callable

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.multiprocessing
import torch.nn.functional as func
from sklearn.metrics import confusion_matrix
from tqdm import tqdm

from configs.types import AudioFeatures, DatasetMode, ADType
from model.manager import Registers
from util.log_util.logger import GlobalLogger
from util.tools.files_util import global_init
from util.train_util.data_loader import AldsDataset
from util.train_util.trainer_util import prepare_dataloader, get_best_acc_weight

torch.multiprocessing.set_sharing_strategy('file_system')


def evaluate_multi_joint(identifier: str, config: Dict, model_list: List[Dict], **kwargs):
    # Get the fold
    total_fold = config['dataset']['k_fold']
    logger = GlobalLogger().get_logger()

    output_all_predict = dict()
    output_all_correct = dict()
    output_all_data = dict()
    output_all_source = dict()
    dict_output = {}

    for index, model_unit in enumerate(model_list):
        model_name = model_unit["model_name"]
        use_features = model_unit["use_features"]
        weight_identifier = model_unit["weight_identifier"]
        input_channels = model_unit["input_channels"]
        joint = model_unit["joint"]
        dataset_func = model_unit["dataset_func"]
        weight_description = model_unit["weight_description"]

        logger.info("Evaluate the model {}".format(model_name))

        output_feature_predict = []
        output_feature_correct = []
        output_feature_source = []
        output_feature_data = []

        total = 0
        correct = 0

        logger.info("Evaluate the model {}.".format(model_name))
        for current_fold, test_dataloader in enumerate(
                prepare_dataloader([use_features] if isinstance(use_features,AudioFeatures) else use_features, config["dataset"], DatasetMode.TEST,
                                   repeat_times=5 * config["dataset"]["repeat_times"]
                                   # repeat_times=1
                    , dataset_func=dataset_func)):

            output_fold_predict = []
            output_fold_correct = []
            output_fold_source = []
            output_fold_data = []

            # Get model and send to GPU
            model = Registers.model[model_name]()
            model = model.cuda()
            # Load weight file into model
            weight_file = get_best_acc_weight(
                os.path.join(config['weight']['weight_dir'], weight_identifier),
                total_fold, current_fold,
                use_features.value if isinstance(use_features, AudioFeatures) else weight_description)
            logger.info("Using weight {}".format(weight_file))
            model.load_state_dict(torch.load(weight_file), strict=True)
            # Set to eval mode
            model.eval()

            # Get the length of the test dataloader
            length = len(test_dataloader)

            # Init the bar
            bar_test = tqdm(range(length))
            bar_test.set_description(
                "Testing using feature {}, for fold {}/{}.".format(use_features, current_fold,
                                                                   total_fold))

            # Re-init the timer
            current_time = time.time()

            with torch.no_grad():
                # Running one batch
                for data in test_dataloader:
                    # Get the features
                    feature_list = []
                    label = data[AudioFeatures.LABEL]
                    file_name = data[AudioFeatures.NAME]
                    output_fold_correct.append(label.numpy())
                    output_fold_source.append(file_name)
                    label = label.cuda()
                    if not joint:
                        features = data[use_features]
                        if input_channels != 1:
                            features = torch.cat([features] * input_channels, dim=1)

                        features = features.cuda()
                        output = model(features)
                    else:
                        for item in use_features:
                            feature = data[item]
                            if input_channels != 1:
                                feature = torch.cat([feature] * input_channels, dim=1)
                            feature = feature.cuda()
                            feature_list.append(feature)
                        output = model(*feature_list)

                    # Normalize the output to one-hot mode
                    predictions = func.softmax(output, dim=1)
                    _, predicted = torch.max(predictions, 1)

                    output_fold_data.append(predictions.cpu().numpy())
                    output_fold_predict.append(predicted.cpu().numpy())

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

                output_feature_correct.append(output_fold_correct)
                output_feature_predict.append(output_fold_predict)
                output_feature_source.append(output_fold_source)
                output_feature_data.append(output_fold_data)
            bar_test.close()
            # Time the timer
            now_time = time.time()
            logger.info(
                "Finish testing feature {}, for fold {}/{}, time cost {:.2f}s ,with acc {:.2f}%".format(
                    use_features.value if isinstance(use_features, AudioFeatures) else weight_description,
                    current_fold,
                    total_fold,
                    now_time - current_time,
                    acc * 100))
        output_all_correct[index] = output_feature_correct
        output_all_predict[index] = output_feature_predict
        output_all_data[index] = output_feature_data
        output_all_source[index] = output_feature_source
    logger.info("All results evaluation finished!")

    output_list = []
    label_list = []
    feature_list = []
    for index, model_unit in enumerate(model_list):
        model_name = model_unit["model_name"]
        use_features = model_unit["use_features"]
        weight_description = model_unit["weight_description"]
        output_dict, label_dict = analysis_output(identifier, config, output_all_correct[index],
                                                  output_all_predict[index],
                                                  output_all_data[index], output_all_source[index],
                                                  "{} with {}".format(model_name,
                                                                      use_features.value if isinstance(use_features,
                                                                                                       AudioFeatures) else weight_description))
        output_list.append(output_dict)
        label_list.append(label_dict)
        feature_list.append(feature)
    files = output_list[0].keys()
    n2s_dict = {0: "AD", 1: "HC", 2: "MCI"}
    s2n_dict = {"AD": 1, "MCI": 2, "HC": 3}
    with open(os.path.join(config["output"]["output_dir"], identifier, "evaluation_on_{}.txt".format("Multi")),
              "w", encoding="utf-8") as fout:
        for file in files:
            fout.write(
                "{} {} ".format(file.split("/")[-1] if "/" in file else file.split("\\")[-1],
                                label_list[0][file]))
            output_data_list = []
            for index, f in enumerate(feature_list):
                output_data = output_list[index][file]
                pred_on_this_f = n2s_dict[int(np.argmax(output_data, axis=0))]
                fout.write("{} {} {} ".format(output_data.tolist(), pred_on_this_f,
                                              pred_on_this_f == label_list[index][file]))
                output_data_list.append(output_data)
            output_average = np.average(np.array(output_data_list), axis=0)
            output_label = n2s_dict[int(np.argmax(output_average))]
            fout.write("-> {} {} {}\n".format(output_average.tolist(), output_label,
                                              output_label == label_list[0][file]))


def evaluate_specific_joint(identifier: str, config: Dict, model_name: str, use_feature_and_weight: Dict,
                            dataset_func: Callable, input_channels: int = 1, **kwargs):
    # Get the fold
    total_fold = config['dataset']['k_fold']
    logger = GlobalLogger().get_logger()

    output_all_predict = dict()
    output_all_correct = dict()
    output_all_data = dict()
    output_all_source = dict()

    dict_output = {}
    for feature in use_feature_and_weight.keys():
        logger.info("Running the feature :{}".format(feature))

        output_feature_predict = []
        output_feature_correct = []
        output_feature_source = []
        output_feature_data = []

        total = 0
        correct = 0

        logger.info("Evaluate the model {}.".format(model_name))
        for current_fold, test_dataloader in enumerate(
                prepare_dataloader([feature], config["dataset"], DatasetMode.TEST,
                                   repeat_times=5 * config["dataset"]["repeat_times"], dataset_func=dataset_func)):

            output_fold_predict = []
            output_fold_correct = []
            output_fold_source = []
            output_fold_data = []

            # Get model and send to GPU
            model = Registers.model[model_name]()
            model = model.cuda()
            # Load weight file into model
            weight_file = get_best_acc_weight(
                os.path.join(config['weight']['weight_dir'], use_feature_and_weight[feature]),
                total_fold, current_fold, feature.value if isinstance(feature, AudioFeatures) else feature)
            logger.info("Using weight {}".format(weight_file))
            model.load_state_dict(torch.load(weight_file), strict=True)
            # Set to eval mode
            model.eval()

            # Get the length of the test dataloader
            length = len(test_dataloader)

            # Init the bar
            bar_test = tqdm(range(length))
            bar_test.set_description(
                "Testing using feature {}, for fold {}/{}.".format(feature, current_fold,
                                                                   total_fold))

            # Re-init the timer
            current_time = time.time()

            with torch.no_grad():
                # Running one batch
                for data in test_dataloader:
                    # Get the features

                    features, label = data[feature], data[AudioFeatures.LABEL]
                    file_name = data[AudioFeatures.NAME]
                    if input_channels != 1:
                        features = torch.cat([features] * input_channels, dim=1)

                    output_fold_correct.append(label.numpy())
                    output_fold_source.append(file_name)

                    features = features.cuda()
                    label = label.cuda()
                    # Running the model
                    output = model(features)
                    # Normalize the output to one-hot mode
                    predictions = func.softmax(output, dim=1)
                    _, predicted = torch.max(predictions, 1)

                    output_fold_data.append(predictions.cpu().numpy())
                    output_fold_predict.append(predicted.cpu().numpy())

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

                output_feature_correct.append(output_fold_correct)
                output_feature_predict.append(output_fold_predict)
                output_feature_source.append(output_fold_source)
                output_feature_data.append(output_fold_data)
            bar_test.close()
            # Time the timer
            now_time = time.time()
            logger.info(
                "Finish testing feature {}, for fold {}/{}, time cost {:.2f}s ,with acc {:.2f}%".format(
                    feature.value if isinstance(feature, AudioFeatures) else feature,
                    current_fold,
                    total_fold,
                    now_time - current_time,
                    acc * 100))
        output_all_correct[feature] = output_feature_correct
        output_all_predict[feature] = output_feature_predict
        output_all_data[feature] = output_feature_data
        output_all_source[feature] = output_feature_source
    logger.info("All results evaluation finished!")
    output_list = []
    label_list = []
    feature_list = []
    for feature in use_feature_and_weight.keys():
        output_dict, label_dict = analysis_output(identifier, config, output_all_correct[feature],
                                                  output_all_predict[feature],
                                                  output_all_data[feature], output_all_source[feature],
                                                  "{} with {}".format(model_name, feature.value))
        output_list.append(output_dict)
        label_list.append(label_dict)
        feature_list.append(feature)
    files = output_list[0].keys()
    n2s_dict = {0: "AD", 1: "HC", 2: "MCI"}
    s2n_dict = {"AD": 1, "MCI": 2, "HC": 3}
    with open(os.path.join(config["output"]["output_dir"], identifier, "evaluation_on_{}.txt".format(model_name)),
              "w", encoding="utf-8") as fout:
        for file in files:
            fout.write(
                "{} {} ".format(file.split("/")[-1] if "/" in file else file.split("\\")[-1],
                                label_list[0][file]))
            output_data_list = []
            for index, f in enumerate(feature_list):
                output_data = output_list[index][file]
                pred_on_this_f = n2s_dict[int(np.argmax(output_data, axis=0))]
                fout.write("{} {} {} ".format(output_data.tolist(), pred_on_this_f,
                                              pred_on_this_f == label_list[index][file]))
                output_data_list.append(output_data)
            output_average = np.average(np.array(output_data_list), axis=0)
            output_label = n2s_dict[int(np.argmax(output_average))]
            fout.write("-> {} {} {}\n".format(output_average.tolist(), output_label,
                                              output_label == label_list[0][file]))


def analysis_output(identifier, config, correct_label, predicted_label, output_data, output_source, extra_info):
    assert len(correct_label) == len(predicted_label) == len(output_data) == len(output_source)
    total_fold = len(correct_label)

    total_correct = []
    total_predicted = []
    total_data = []
    total_files = []
    for fold in range(total_fold):
        correct = np.concatenate(correct_label[fold])
        total_correct.append(correct)
        predicted = np.concatenate(predicted_label[fold])
        total_predicted.append(predicted)
        data_out = np.concatenate(output_data[fold])
        total_data.append(data_out)
        file_out = np.concatenate(output_source[fold])
        total_files.append(file_out)

        acc = len(correct[correct == predicted]) / len(correct)

        matrix = confusion_matrix(y_true=correct, y_pred=predicted)
        plot_image(identifier, config, matrix, [i.value for i in ADType],
                   "{} {}-{} Fold Results\nAccuracy {:.2f} Percent".format(extra_info, fold, total_fold, acc * 100))
        logger.info("Finish generating image {}-{} for {}.".format(fold, total_fold, extra_info))
    correct = np.concatenate(total_correct)
    predicted = np.concatenate(total_predicted)
    data = np.concatenate(total_data)
    files = np.concatenate(total_files)

    keys = list(set(files))
    list.sort(keys)
    output_dict = {}
    label_dict = {}
    n2s_dict = {0: "AD", 1: "HC", 2: "MCI"}
    s2n_dict = {"AD": 1, "MCI": 2, "HC": 3}
    for key in keys:
        labels = correct[files == key]
        assert labels.mean() == labels[0]
        label = n2s_dict[labels[0]]
        predictions = np.average(data[files == key], axis=0)
        predict_label = n2s_dict[int(np.argmax(predictions))]
        output_dict[key] = predictions
        label_dict[key] = label

    matrix = confusion_matrix(y_true=correct, y_pred=predicted)
    acc = len(correct[correct == predicted]) / len(correct)
    plot_image(identifier, config, matrix, [i.value for i in ADType],
               "{} Results\nAccuracy {:.2f} Percent".format(extra_info, acc * 100))
    logger.info("Finish generating all images for {}.".format(extra_info))
    return output_dict, label_dict


def evaluate_specific(identifier: str, config: Dict, model_name: str, use_feature: Union[AudioFeatures, str],
                      weight_identifier: str, dataset_func: Callable, input_channels: int = 1, **kwargs):
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
                   weight_identifier: str, weight_description: str, dataset_func: Callable, input_channels: int = 1,
                   **kwargs):
    correct_label = []
    predicted_label = []
    total = 0
    correct = 0

    # Get the fold
    total_fold = config['dataset']['k_fold']
    logger = GlobalLogger().get_logger()
    logger.info("Evaluate the model {}.".format(model_name))
    for current_fold, test_dataloader in enumerate(
            prepare_dataloader(use_feature, config["dataset"], DatasetMode.TEST, dataset_func=dataset_func,
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


def analysis_result(identifier, config, correct_label, predicted_label, extra_info):
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
                   "{} {}-{} Fold Results\nAccuracy {:.2f} Percent".format(extra_info, fold, total_fold, acc * 100))
        logger.info("Finish generating image {}-{}.".format(fold, total_fold))
    correct = np.concatenate(total_correct)
    predicted = np.concatenate(total_predicted)
    matrix = confusion_matrix(y_true=correct, y_pred=predicted)
    acc = len(correct[correct == predicted]) / len(correct)
    plot_image(identifier, config, matrix, [i.value for i in ADType],
               "{} Results\nAccuracy {:.2f} Percent".format(extra_info, acc * 100))
    logger.info("Finish generating all images.")


if __name__ == '__main__':
    time_identifier, configs = global_init(True)
    logger = GlobalLogger().get_logger()
    # model_name = "SpecificTrainLongModel"
    # weight_identifier = "20210905_133648"
    # c, p = evaluate_joint(time_identifier, configs, model_name,
    #                       [AudioFeatures.MFCC, AudioFeatures.SPECS, AudioFeatures.MELSPECS], weight_identifier,
    #                       "Fine_tune", input_shape=())
    # c, p = evaluate_specific(time_identifier, configs, model_name,
    #                          AudioFeatures.MFCC, weight_identifier, AldsDataset, input_channels=3)
    # logger.info("Analysis results for {} with {}".format(model_name, weight_identifier))
    # analysis_result(time_identifier, configs, c, p, model_name)
    # model_name = "CompetitionSpecificTrainVggNet19BNBackboneModel"
    # weight_identifier = "20210905_133648"
    # evaluate_specific_joint(time_identifier, configs, model_name,
    #                         {AudioFeatures.MFCC: "20210915_184216"
    #                             , AudioFeatures.MELSPECS: "20210915_183611"
    #                             , AudioFeatures.SPECS: "20210915_183922"},
    #                         AldsDataset, input_shape=(), input_channels=3)
    model_list = [
        {"model_name": "SpecificTrainLongModel",
         "use_features": AudioFeatures.MFCC,
         "weight_identifier": "20210905_133648",
         "input_channels": 1,
         "joint": False,
         "dataset_func": AldsDataset,
         "weight_description": ""},
        {"model_name": "SpecificTrainLongModel",
         "use_features": AudioFeatures.MELSPECS,
         "weight_identifier": "20210905_133648",
         "input_channels": 1,
         "joint": False,
         "dataset_func": AldsDataset,
         "weight_description": ""},
        {"model_name": "MSMJointConcatFineTuneLongModel",
         "use_features": [AudioFeatures.MFCC, AudioFeatures.SPECS, AudioFeatures.MELSPECS],
         "weight_identifier": "20210907_230640",
         "input_channels": 1,
         "joint": True,
         "dataset_func": AldsDataset,
         "weight_description": "General"}
    ]
    evaluate_multi_joint(time_identifier, configs, model_list)
