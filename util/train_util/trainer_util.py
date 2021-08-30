import os
from typing import List, Dict, Generic, Union, Optional, Callable
from torch.utils.data.dataloader import DataLoader
from configs.types import AudioFeatures, DatasetMode
from util.log_util.logger import GlobalLogger
from util.tools.files_util import create_dir
from util.train_util.data_loader import AldsDataset2D
import numpy as np
from torch import nn
from torch import optim
from network.general.general_model import GeneralModel, ConvModel
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


def prepare_dataloader(use_features: List[AudioFeatures], dataset_fn: Callable, configs: Dict, run_for: DatasetMode,
                       **kwargs):
    """
    This function returns the generator of dataloader.
    Considering the k-fold is used in the program so the function is design to be the generator.
    Notice that even the k-fold is not used the function will still return a dataloader generator with the only one dataloader.
    :param use_features: List[AudioFeatures], the list of AudioFeatures and determines which features are used in dataset.
    :param dataset_fn: Callable function for Dataset instance.
    :param configs: Dict, and should be configs['dataset'] by default
    :param run_for: DatasetMode, this parameters in used to determine what the dataset aims to.
    :param kwargs: other parameters, notice that any given parameters will override the parameters in config
    :return: generators of dataloader, the length is same as the k_fold
    """
    # override the parameters in configs if given in kwargs
    use_merge = configs['use_merge'] if 'use_merge' not in kwargs.keys() else kwargs['use_merge']
    repeat_times = configs['repeat_times'] if 'repeat_times' not in kwargs.keys() else kwargs['repeat_times']
    k_fold = configs['k_fold'] if 'k_fold' not in kwargs.keys() else kwargs['k_fold']
    batch_size = configs['batch_size'] if 'batch_size' not in kwargs.keys() else kwargs['batch_size']
    random_disruption = configs['random_disruption'] if 'random_disruption' not in kwargs.keys() else kwargs[
        'random_disruption']
    if k_fold != 0:
        # Generate the k_fold dataloader
        for fold in range(k_fold):
            dataset = dataset_fn(use_features=use_features, use_merge=use_merge,
                                 repeat_times=repeat_times, configs=configs['process'], k_fold=k_fold,
                                 current_fold=fold, random_disruption=random_disruption,
                                 run_for=run_for)

            dataloader = DataLoader(dataset, batch_size=batch_size)
            yield dataloader
    else:
        # Generate the single dataloader
        for fold in range(1):
            dataset = dataset_fn(use_features=use_features, use_merge=use_merge,
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


def train_specific_feature(configs: Dict, time_identifier: str, specific_feature: AudioFeatures,
                           dataset_func: Callable, model_func: Callable, *args, **kwargs) -> None:
    """
    This is the trainer of training only with one specific features.
    :param configs: Dict, the configs
    :param time_identifier: str, the global identifier
    :param specific_feature: AudioFeatures, the feature to use
    :param dataset_func: function call, the custom dataset
    :param model_func: function call, the custom model
    :return: None
    """
    # Init the saving directory
    save_dir = os.path.join(configs['weight']['weight_dir'], time_identifier, specific_feature.value)
    create_dir(save_dir)
    # Get the fold
    total_fold = configs['dataset']['k_fold']
    # Get logger
    logger = GlobalLogger().get_logger()

    # Getting the dataloader from the generator
    for current_fold, (train_dataloader, test_dataloader) in enumerate(
            zip(prepare_dataloader([specific_feature], dataset_func, configs["dataset"], DatasetMode.TRAIN),
                prepare_dataloader([specific_feature], dataset_func, configs["dataset"], DatasetMode.TEST))):
        # Getting the epoch
        epoch = configs['train']['epoch']

        # If not running on GPU
        model = model_func(*args, **kwargs)
        model = model.cuda()

        # Init the criterion, CE by default
        criterion = nn.CrossEntropyLoss()
        # Init the optimizer, SGD by default
        optimizer = optim.Adam(model.parameters(), lr=0.002)

        for current_epoch in range(1, epoch + 1):
            # Setting the model to train mode
            model.train()
            # Get the length of the dataloader
            length = len(train_dataloader)
            # Init the loss
            running_loss = 0.0
            # Init the timer
            current_time = time.time()
            # Create the tqdm bar
            bar = tqdm(range(length))
            bar.set_description(
                "Training using feature {}, for fold {}/{}, epoch {}".format(specific_feature, current_fold,
                                                                             total_fold,
                                                                             current_epoch))
            # Running one batch
            for iteration, data in enumerate(train_dataloader):
                feature, label = data[0], data[-1]
                # Get features and set them to cuda
                feature = feature.cuda()
                label = label.cuda()
                # Set the optimizer to zero
                optimizer.zero_grad()
                # Go through one epoch
                output = model(feature)
                # Calculate the loss
                loss = criterion(output, label)
                # Back propagation
                loss.backward()
                # Update the optimizer
                optimizer.step()
                # Sum up the losses
                running_loss += loss.item()
                # Visualize the loss
                bar.set_postfix(loss=running_loss / (iteration + 1))
                # Update the bar
                bar.update(1)
            # Training finish, close the bar
            bar.close()
            # Calculate the final loss
            losses = running_loss / length
            # Time the time past
            now_time = time.time()
            # Write logs
            logger.info(
                "Finish training feature {}, for fold {}/{}, epoch {}, time cost {}s ,with loss {}".format(
                    specific_feature,
                    current_fold,
                    total_fold,
                    current_epoch,
                    now_time - current_time,
                    losses))
            # Re-init the timer
            current_time = time.time()
            # Going into eval mode
            correct = 0
            total = 0
            # Get the length of the test dataloader
            length = len(test_dataloader)
            # Init the bar
            bar_test = tqdm(range(length))
            bar_test.set_description(
                "Testing using feature {}, for fold {}/{}, epoch {}".format(specific_feature, current_fold,
                                                                            total_fold,
                                                                            current_epoch))
            # Set the model to evaluation mode
            model.eval()
            # Do not record the gradiant
            with torch.no_grad():
                # Running one batch
                for data in test_dataloader:
                    # Get the features
                    feature, label = data[0], data[-1]
                    feature = feature.cuda()
                    label = label.cuda()
                    # Running the model
                    output = model(feature)
                    # Normalize the output to one-hot mode
                    _, predicted = torch.max(func.softmax(output, dim=1), 1)
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
            # Calculate the accuracy
            final = correct / total
            # Close the bar
            bar_test.close()
            # Time the timer
            now_time = time.time()
            # Write the log
            logger.info(
                "Finish testing feature {}, for fold {}/{}, epoch {}, time cost {}s ,with acc {}".format(
                    specific_feature,
                    current_fold,
                    total_fold,
                    current_epoch,
                    now_time - current_time,
                    final))
            # Save the weight to the directory
            save_name = os.path.join(save_dir, "fold{}_{}-epoch{}-loss{}-acc{}.pth").format(current_fold,
                                                                                            total_fold,
                                                                                            current_epoch, losses,
                                                                                            final)
            torch.save(model.state_dict(), save_name)
            # Write the log
            logger.info("Saving weight to {}".format(save_name))


def train_general(configs: Dict, time_identifier: str, use_features: List[AudioFeatures],
                  fine_tune: bool = False, load_weight_identifier: Optional[str] = None,
                  weighted_dir: Optional[str] = None, *args, **kwargs) -> None:
    """
    This is the trainer of training with joint-features.
    :param configs: Dict, the configs
    :param time_identifier: str, the global identifier
    :param use_features: List[AudioFeatures], the feature to use
    :param fine_tune: bool, whether to fine-tune the model
    :param load_weight_identifier: str, if given, the weight will load from the given directory instead of the time_identifier
    :param weighted_dir: str, the name of the directory to load weight from
    :return: None
    """
    # Check GPU device
    device = 0 if "cuda" not in kwargs.keys() else kwargs["cuda"]
    # Init the logger
    logger = GlobalLogger().get_logger()
    logger.info("Training the general model.")
    # Init the saving directory
    if not fine_tune:
        save_dir = os.path.join(configs['weight']['weight_dir'], time_identifier, "General")
    else:
        save_dir = os.path.join(configs['weight']['weight_dir'], time_identifier, "Fine_tune")
    create_dir(save_dir)
    # Get the fold
    total_fold = configs['dataset']['k_fold']
    # Getting the dataloader from the generator
    for current_fold, (train_dataloader, test_dataloader) in enumerate(
            zip(prepare_dataloader(use_features, AldsDataset2D, configs["dataset"], DatasetMode.TRAIN),
                prepare_dataloader(use_features, AldsDataset2D, configs["dataset"], DatasetMode.TEST))):
        # Getting the epoch
        epoch = configs['train']['epoch']
        # Send the model to GPU
        model = GeneralModel()
        model.cuda(device)

        # Init the criterion, CE by default
        criterion = nn.CrossEntropyLoss()
        # Init the optimizer, SGD by default
        optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

        # Get the weight
        weight_identifier = time_identifier
        # if the load_weight_identifier not given, time_identifier will be used
        if load_weight_identifier is not None:
            weight_identifier = load_weight_identifier

        # If common training
        if not fine_tune:
            # Load weight from all separated directories
            for specific_feature in use_features:
                # By default the bast accuracy weight is used
                weight_file = get_best_acc_weight(os.path.join(configs['weight']['weight_dir'], weight_identifier),
                                                  total_fold, current_fold, specific_feature)
                # Load weights
                if specific_feature == AudioFeatures.SPECS:
                    model.extractor_spec.load_state_dict(torch.load(weight_file), strict=False)
                if specific_feature == AudioFeatures.MELSPECS:
                    model.extractor_mel.load_state_dict(torch.load(weight_file), strict=False)
                if specific_feature == AudioFeatures.MFCC:
                    model.extractor_mfcc.load_state_dict(torch.load(weight_file), strict=False)
                # Write the logs
                logger.info("Load weight {} for {}.".format(weight_file, specific_feature.value))
        # Fine-tune the model, and the weight should be in one directory and do not need to be loaded separately
        else:
            # If weighted_dir not given, reset ot 'General' by default
            if weighted_dir is None:
                logger.warning("weighted_dir is None, set to 'General'.")
                weighted_dir = "General"
            # By default the bast accuracy weight is used
            weight_file = get_best_acc_weight(os.path.join(configs['weight']['weight_dir'], weight_identifier),
                                              total_fold, current_fold, weighted_dir)
            # Load weights
            model.load_state_dict(torch.load(weight_file), strict=True)
            # Write the logs
            logger.info("Load weight {} for fine-tuning.".format(weight_file))
        # If common training, the parameters in extractor should be frozen
        if not fine_tune:
            for param in model.extractor_spec.parameters():
                param.requires_grad = False
            for param in model.extractor_mfcc.parameters():
                param.requires_grad = False
            for param in model.extractor_mel.parameters():
                param.requires_grad = False
            logger.info("In mode training, freeze the extractor layers.")
        # If fine-tune, the parameters in extractor should be unfreeze
        else:
            for param in model.extractor_spec.parameters():
                param.requires_grad = True
            for param in model.extractor_mfcc.parameters():
                param.requires_grad = True
            for param in model.extractor_mel.parameters():
                param.requires_grad = True
            logger.info("In mode fine-tune, unfreeze the extractor layers.")
        # Running epoch
        for current_epoch in range(1, epoch + 1):
            # Setting the model to train mode
            model.train()
            # Get the length of the dataloader
            length = len(train_dataloader)
            # Init the loss
            running_loss = 0.0
            # Init the timer
            current_time = time.time()
            # Create the tqdm bar
            bar = tqdm(range(length))
            if not fine_tune:
                bar.set_description(
                    "Training general model with frozen layers, for fold {}/{}, epoch {}".format(current_fold,
                                                                                                 total_fold,
                                                                                                 current_epoch))
            else:
                bar.set_description(
                    "Fine-tuning general model, for fold {}/{}, epoch {}".format(current_fold, total_fold,
                                                                                 current_epoch))
            # Running one batch
            for iteration, data in enumerate(train_dataloader):
                # Get features and set them to cuda
                spec, mel, mfcc, label = data[use_features.index(AudioFeatures.SPECS)], data[
                    use_features.index(AudioFeatures.MELSPECS)], data[use_features.index(AudioFeatures.MFCC)], data[-1]
                spec = spec.cuda(device)
                mel = mel.cuda(device)
                mfcc = mfcc.cuda(device)
                label = label.cuda(device)
                # Set the optimizer to zero
                optimizer.zero_grad()
                # Go through one epoch
                output = model(spec, mel, mfcc)
                # Calculate the loss
                loss = criterion(output, label)
                # Back propagation
                loss.backward()
                # Update the optimizer
                optimizer.step()
                # Sum up the losses
                running_loss += loss.item()
                # Visualize the loss
                bar.set_postfix(loss=running_loss / (iteration + 1))
                # Update the bar
                bar.update(1)
            # Training finish, close the bar
            bar.close()
            # Calculate the final loss
            losses = running_loss / length
            # Time the time past
            now_time = time.time()
            # Write logs
            if not fine_tune:
                logger.info(
                    "Finish training general model, for fold {}/{}, epoch {}, time cost {}s ,with loss {}".format(
                        current_fold,
                        total_fold,
                        current_epoch,
                        now_time - current_time,
                        losses))
            else:
                logger.info(
                    "Finish fine-tune general model, for fold {}/{}, epoch {}, time cost {}s ,with loss {}".format(
                        current_fold,
                        total_fold,
                        current_epoch,
                        now_time - current_time,
                        losses))
            # Re-init the timer
            current_time = time.time()
            # Going into eval mode
            correct = 0
            total = 0
            # Get the length of the test dataloader
            length = len(test_dataloader)
            # Init the bar
            bar_test = tqdm(range(length))
            bar_test.set_description(
                "Testing general model, for fold {}/{}, epoch {}".format(current_fold,
                                                                         total_fold,
                                                                         current_epoch))
            # Set the model to evaluation mode
            model.eval()
            # Do not record the gradiant
            with torch.no_grad():
                # Running one batch
                for data in test_dataloader:
                    # Get the features
                    spec, mel, mfcc, label = data[use_features.index(AudioFeatures.SPECS)], data[
                        use_features.index(AudioFeatures.MELSPECS)], data[use_features.index(AudioFeatures.MFCC)], data[
                                                 -1]
                    spec = spec.cuda(device)
                    mel = mel.cuda(device)
                    mfcc = mfcc.cuda(device)
                    label = label.cuda(device)
                    # Running the model
                    output = model(spec, mel, mfcc)
                    # Normalize the output to one-hot mode
                    _, predicted = torch.max(func.softmax(output, dim=1), 1)
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
            # Calculate the accuracy
            final = correct / total
            # Close the bar
            bar_test.close()
            # Time the timer
            now_time = time.time()
            # Write the log
            logger.info(
                "Finish testing general model, for fold {}/{}, epoch {}, time cost {}s ,with acc {}".format(
                    current_fold,
                    total_fold,
                    current_epoch,
                    now_time - current_time,
                    final))
            # Save the weight to the directory
            save_name = os.path.join(save_dir, "fold{}_{}-epoch{}-loss{}-acc{}.pth").format(current_fold,
                                                                                            total_fold,
                                                                                            current_epoch, losses,
                                                                                            final)
            torch.save(model.state_dict(), save_name)
            # Write the log
            logger.info("Saving weight to {}".format(save_name))
