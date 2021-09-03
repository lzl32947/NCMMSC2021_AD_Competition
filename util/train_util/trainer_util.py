import os
from typing import List, Dict, Generic, Union, Optional
from torch.utils.data.dataloader import DataLoader
from configs.types import AudioFeatures, DatasetMode
from util.log_util.logger import GlobalLogger
from util.tools.files_util import create_dir
from util.train_util.data_loader import AldsDataset
import numpy as np
from torch import nn
from torch import optim
from tqdm import tqdm
import time
import torch
import torch.nn.functional as func


def prepare_feature(feature_list: List[str]) -> List[AudioFeatures]:
    """
    This function is to get which features are used in dataset.
    :param feature_list: List[str], and should be the configs['features']
    :return: list[AudioFeatures]
    """
    use_features = []
    # Traverse the feature list to convert them into AudioFeatures
    for feature in feature_list:
        for item in AudioFeatures:
            if item.value == feature:
                use_features.append(item)
    return use_features


def prepare_dataloader(use_features: List[AudioFeatures], configs: Dict, run_for: DatasetMode, **kwargs):
    """
    This function returns the generator of dataloader.
    Considering the k-fold is used in the program so the function is design to be the generator.
    Notice that even the k-fold is not used the function will still return a dataloader generator with the only one dataloader.
    :param use_features: List[AudioFeatures], the list of AudioFeatures and determines which features are used in dataset.
    :param configs: Dict, and should be configs['dataset'] by default
    :param run_for: DatasetMode, this parameters in used to determine what the dataset aims to.
    :param kwargs: other parameters, notice that any given parameters will override the parameters in config
    :return: generators of dataloader, the length is same as the k_fold
    """
    # override the parameters in configs if given in kwargs
    use_merge = configs['use_merge'] if 'use_merge' not in kwargs.keys() else kwargs['use_merge']
    use_vad = configs['use_vad'] if 'use_vad' not in kwargs.keys() else kwargs['use_vad']
    repeat_times = configs['repeat_times'] if 'repeat_times' not in kwargs.keys() else kwargs['repeat_times']
    k_fold = configs['k_fold'] if 'k_fold' not in kwargs.keys() else kwargs['k_fold']
    batch_size = configs['batch_size'] if 'batch_size' not in kwargs.keys() else kwargs['batch_size']
    random_disruption = configs['random_disruption'] if 'random_disruption' not in kwargs.keys() else kwargs[
        'random_disruption']
    if k_fold != 0:
        # Generate the k_fold dataloader
        for fold in range(k_fold):
            dataset = AldsDataset(use_features=use_features, use_merge=use_merge, use_vad=use_vad,
                                  repeat_times=repeat_times, configs=configs['process'], k_fold=k_fold,
                                  current_fold=fold, random_disruption=random_disruption,
                                  run_for=run_for)

            dataloader = DataLoader(dataset, batch_size=batch_size)
            yield dataloader
    else:
        # Generate the single dataloader
        for fold in range(1):
            dataset = AldsDataset(use_features=use_features, use_merge=use_merge,use_vad=use_vad,
                                  repeat_times=repeat_times, configs=configs['process'],
                                  random_disruption=random_disruption,
                                  run_for=run_for)

            dataloader = DataLoader(dataset, batch_size=batch_size)
            yield dataloader


def read_weight(weight_dir: str, specific_feature: Union[AudioFeatures, str]) -> (
        List[str], List[str], List[str], List[str], List[str], List[str]):
    """
    Load all the weight files to the separated lists.
    :param weight_dir: str, the path to the weight directory, NOTICE that this directory is the direct directory that contains the features directory
    :param specific_feature: AudioFeatures of str, AudioFeatures will be cast to AudioFeatures.value and should be the directory name of weight
    :return: List[str], List[str], List[str], List[str], List[str], List[str], separately represents the path to weight files,
            the current fold list, the total fold list, the current epoch list, the loss list and the accuracy list
    """
    # Get the directory that contain the weight files
    if isinstance(specific_feature, AudioFeatures):
        directory = os.path.join(weight_dir, specific_feature.value)
    else:
        directory = os.path.join(weight_dir, specific_feature)
    # Init the list
    file_list, current_list, total_list, epoch_list, loss_list, acc_list = [], [], [], [], [], []
    # Traverse the directory
    for files in os.listdir(directory):
        file_list.append(os.path.join(directory, files))
        # Split the file name
        filename = files.split(".pth")[0]
        # Split the fold, epoch, loss and accuracy
        fold, epoch, loss, acc = filename.split("-")
        # 'fold' in format of 'fold{}_{}'
        current_fold = int(fold[4:].split("_")[0])
        current_list.append(current_fold)
        total_fold = int(fold[4:].split("_")[1])
        total_list.append(total_fold)
        # 'epoch' in format of 'epoch{}'
        epoch = int(epoch[5:])
        epoch_list.append(epoch)
        # 'loss' in format of 'loss{}'
        loss = float(loss[4:])
        loss_list.append(loss)
        # 'accuracy' in format of 'acc{}'
        acc = float(acc[3:])
        acc_list.append(acc)
    # Return the lists
    return file_list, current_list, total_list, epoch_list, loss_list, acc_list


def get_best_acc_weight(weight_dir: str, fold: int, current_fold: int,
                        specific_feature: Union[AudioFeatures, str]) -> str:
    """
    Get the weight file that has the best accuracy in this fold
    :param weight_dir: str, the path to the weight directory, NOTICE that this directory is the direct directory that contains the features directory
    :param fold: int, the total fold
    :param current_fold: int, the current fold
    :param specific_feature: AudioFeatures of str, AudioFeatures will be cast to AudioFeatures.value and should be the directory name of weight
    :return: str, the path to weight file
    """
    # Read the information from the weight directory
    file_list, current_list, total_list, epoch_list, loss_list, acc_list = read_weight(weight_dir, specific_feature)
    # Transform the folds into np.ndarray for comparing
    total_list = np.array(total_list)
    # ALL THE FOLD SHOULD BE THE SAME
    assert fold == total_list.mean()
    # Get the current list and transform to np.ndarray
    current_list = np.array(current_list)
    acc_list = np.array(acc_list)
    # Get the index of the best accuracy file
    acc_index = np.argmax(acc_list[current_list == current_fold])
    # Return the path to that file
    return file_list[acc_index]


def get_best_loss_weight(weight_dir: str, fold: int, current_fold: int,
                         specific_feature: Union[AudioFeatures, str]) -> str:
    """
    Get the weight file that has the least loss in this fold
    :param weight_dir: str, the path to the weight directory, NOTICE that this directory is the direct directory that contains the features directory
    :param fold: int, the total fold
    :param current_fold: int, the current fold
    :param specific_feature: AudioFeatures of str, AudioFeatures will be cast to AudioFeatures.value and should be the directory name of weight
    :return: str, the path to weight file
    """
    # Read the information from the weight directory
    file_list, current_list, total_list, epoch_list, loss_list, acc_list = read_weight(weight_dir, specific_feature)
    # Transform the folds into np.ndarray for comparing
    total_list = np.array(total_list)
    # ALL THE FOLD SHOULD BE THE SAME
    assert fold == total_list.mean()
    # Get the current list and transform to np.ndarray
    current_list = np.array(current_list)
    acc_list = np.array(loss_list)
    # Get the index of the least loss file
    acc_index = np.argmin(acc_list[current_list == current_fold])
    # Return the path to that file
    return file_list[acc_index]

