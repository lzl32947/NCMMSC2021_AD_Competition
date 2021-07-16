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
    repeat_times = configs['repeat_times'] if 'repeat_times' not in kwargs.keys() else kwargs['repeat_times']
    k_fold = configs['k_fold'] if 'k_fold' not in kwargs.keys() else kwargs['k_fold']
    batch_size = configs['batch_size'] if 'batch_size' not in kwargs.keys() else kwargs['batch_size']
    random_disruption = configs['random_disruption'] if 'random_disruption' not in kwargs.keys() else kwargs[
        'random_disruption']
    if k_fold != 0:
        # Generate the k_fold dataloader
        for fold in range(k_fold):
            dataset = AldsDataset(use_features=use_features, use_merge=use_merge,
                                  repeat_times=repeat_times, configs=configs['process'], k_fold=k_fold,
                                  current_fold=fold, random_disruption=random_disruption,
                                  run_for=run_for)

            dataloader = DataLoader(dataset, batch_size=batch_size)
            yield dataloader
    else:
        # Generate the single dataloader
        for fold in range(1):
            dataset = AldsDataset(use_features=use_features, use_merge=use_merge,
                                  repeat_times=repeat_times, configs=configs['process'],
                                  random_disruption=random_disruption,
                                  run_for=run_for)

            dataloader = DataLoader(dataset, batch_size=batch_size)
            yield dataloader


def read_weight(weight_dir: str, specific_feature: Union[AudioFeatures, str]):
    if isinstance(specific_feature, AudioFeatures):
        directory = os.path.join(weight_dir, specific_feature.value)
    else:
        directory = os.path.join(weight_dir, specific_feature)

    file_list, current_list, total_list, epoch_list, loss_list, acc_list = [], [], [], [], [], []

    for files in os.listdir(directory):
        file_list.append(os.path.join(weight_dir, specific_feature.value, files))
        filename = files.split(".pth")[0]
        fold, epoch, loss, acc = filename.split("-")
        current_fold = int(fold[4:].split("_")[0])
        current_list.append(current_fold)
        total_fold = int(fold[4:].split("_")[1])
        total_list.append(total_fold)
        epoch = int(epoch[5:])
        epoch_list.append(epoch)
        loss = float(loss[4:])
        loss_list.append(loss)
        acc = float(acc[3:])
        acc_list.append(acc)
    return file_list, current_list, total_list, epoch_list, loss_list, acc_list


def get_best_acc_weight(weight_dir: str, fold: int, current_fold: int, specific_feature: Union[AudioFeatures, str]):
    file_list, current_list, total_list, epoch_list, loss_list, acc_list = read_weight(weight_dir, specific_feature)
    total_list = np.array(total_list)
    assert fold == total_list.mean()
    current_list = np.array(current_list)
    acc_list = np.array(acc_list)
    acc_index = np.argmax(acc_list[current_list == current_fold])
    return file_list[acc_index]


def train_specific_feature(configs: Dict, time_identifier: str, specific_feature: AudioFeatures, model: nn.Module):
    save_dir = os.path.join(configs['weight']['weight_dir'], time_identifier, specific_feature.value)
    total_fold = configs['dataset']['k_fold']
    create_dir(save_dir)
    logger = GlobalLogger().get_logger()

    for current_fold, (train_dataloader, test_dataloader) in enumerate(
            zip(prepare_dataloader([specific_feature], configs["dataset"], DatasetMode.TRAIN),
                prepare_dataloader([specific_feature], configs["dataset"], DatasetMode.TEST))):
        epoch = configs['train']['epoch']

        if not model.cuda:
            model = model.cuda()

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

        for current_epoch in range(1, epoch + 1):
            model.train()
            length = len(train_dataloader)
            running_loss = 0.0
            current_time = time.time()

            bar = tqdm(range(length))
            bar.set_description(
                "Training using feature {}, for fold {}/{}, epoch {}".format(specific_feature, current_fold,
                                                                             total_fold,
                                                                             current_epoch))
            for iteration, data in enumerate(train_dataloader):
                feature, label = data[0], data[-1]

                feature = feature.cuda()
                label = label.cuda()

                optimizer.zero_grad()
                output = model(feature)

                loss = criterion(output, label)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()

                bar.set_postfix(loss=running_loss / (iteration + 1))
                bar.update(1)
            bar.close()
            losses = running_loss / length
            now_time = time.time()
            logger.info(
                "Finish training feature {}, for fold {}/{}, epoch {}, time cost {}s ,with loss {}".format(
                    specific_feature,
                    current_fold,
                    total_fold,
                    current_epoch,
                    now_time - current_time,
                    losses))
            current_time = time.time()

            correct = 0
            total = 0
            length = len(test_dataloader)
            bar_test = tqdm(range(length))
            bar_test.set_description(
                "Testing using feature {}, for fold {}/{}, epoch {}".format(specific_feature, current_fold,
                                                                            total_fold,
                                                                            current_epoch))
            model.eval()
            with torch.no_grad():
                for data in test_dataloader:
                    feature, label = data[0], data[-1]
                    feature = feature.cuda()
                    label = label.cuda()

                    output = model(feature)
                    _, predicted = torch.max(func.softmax(output, dim=1), 1)
                    total += label.size(0)
                    correct += (predicted == label).sum().item()
                    acc = correct / total
                    bar_test.set_postfix(acc=acc)
                    bar_test.update(1)
            final = correct / total
            bar_test.close()
            now_time = time.time()
            logger.info(
                "Finish testing feature {}, for fold {}/{}, epoch {}, time cost {}s ,with acc {}".format(
                    specific_feature,
                    current_fold,
                    total_fold,
                    current_epoch,
                    now_time - current_time,
                    final))
            save_name = os.path.join(save_dir, "fold{}_{}-epoch{}-loss{}-acc{}.pth").format(current_fold,
                                                                                            total_fold,
                                                                                            current_epoch, losses,
                                                                                            final)
            torch.save(model.state_dict(), save_name)
            logger.info("Saving weight to {}".format(save_name))


def train_general(configs: Dict, time_identifier: str, use_features: List[AudioFeatures],
                  fine_tune: bool = False,
                  load_weight_identifier: Optional[str] = None, weighted_dir: Optional[str] = None):
    logger = GlobalLogger().get_logger()
    logger.info("Training the general model.")
    if not fine_tune:
        save_dir = os.path.join(configs['weight']['weight_dir'], time_identifier, "General")
    else:
        save_dir = os.path.join(configs['weight']['weight_dir'], time_identifier, "Fine_tune")
    create_dir(save_dir)
    total_fold = configs['dataset']['k_fold']
    for current_fold, (train_dataloader, test_dataloader) in enumerate(
            zip(prepare_dataloader(use_features, configs["dataset"], DatasetMode.TRAIN),
                prepare_dataloader(use_features, configs["dataset"], DatasetMode.TEST))):
        epoch = configs['train']['epoch']
        model = GeneralModel()
        model.cuda()

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

        weight_identifier = time_identifier
        if load_weight_identifier is not None:
            weight_identifier = load_weight_identifier

        if not fine_tune:
            for specific_feature in use_features:
                weight_file = get_best_acc_weight(os.path.join(configs['weight']['weight_dir'], weight_identifier),
                                                  total_fold, current_fold, specific_feature)
                if specific_feature == AudioFeatures.SPECS:
                    model.extractor_spec.load_state_dict(torch.load(weight_file), strict=False)
                if specific_feature == AudioFeatures.MELSPECS:
                    model.extractor_mel.load_state_dict(torch.load(weight_file), strict=False)
                if specific_feature == AudioFeatures.MFCC:
                    model.extractor_mfcc.load_state_dict(torch.load(weight_file), strict=False)
                logger.info("Load weight {} for {}.".format(weight_file, specific_feature.value))
        else:
            if weighted_dir is None:
                logger.warning("weighted_dir is None, set to 'General'.")
                weighted_dir = "General"
            weight_file = get_best_acc_weight(os.path.join(configs['weight']['weight_dir'], weight_identifier),
                                              total_fold, current_fold, weighted_dir)
            model.load_state_dict(torch.load(weight_file), strict=True)
            logger.info("Load weight {} for fine-tuning.".format(weight_file))
        if not fine_tune:
            for param in model.extractor_spec.parameters():
                param.requires_grad = False
            for param in model.extractor_mfcc.parameters():
                param.requires_grad = False
            for param in model.extractor_mel.parameters():
                param.requires_grad = False
            logger.info("In mode training, freeze the extractor layers.")
        else:
            for param in model.extractor_spec.parameters():
                param.requires_grad = True
            for param in model.extractor_mfcc.parameters():
                param.requires_grad = True
            for param in model.extractor_mel.parameters():
                param.requires_grad = True
            logger.info("In mode fine-tune, unfreeze the extractor layers.")
        for current_epoch in range(1, epoch + 1):
            model.train()

            length = len(train_dataloader)
            running_loss = 0.0
            current_time = time.time()

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
            for iteration, data in enumerate(train_dataloader):
                spec, mel, mfcc, label = data[use_features.index(AudioFeatures.SPECS)], data[
                    use_features.index(AudioFeatures.MELSPECS)], data[use_features.index(AudioFeatures.MFCC)], data[-1]
                spec = spec.cuda()
                mel = mel.cuda()
                mfcc = mfcc.cuda()
                label = label.cuda()

                optimizer.zero_grad()

                output = model(spec, mel, mfcc)

                loss = criterion(output, label)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                bar.set_postfix(loss=running_loss / (iteration + 1))
                bar.update(1)
            bar.close()
            losses = running_loss / length
            now_time = time.time()
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
            current_time = time.time()

            correct = 0
            total = 0
            length = len(test_dataloader)
            bar_test = tqdm(range(length))
            bar_test.set_description(
                "Testing general model, for fold {}/{}, epoch {}".format(current_fold,
                                                                         total_fold,
                                                                         current_epoch))
            model.eval()
            with torch.no_grad():
                for data in test_dataloader:
                    spec, mel, mfcc, label = data[use_features.index(AudioFeatures.SPECS)], data[
                        use_features.index(AudioFeatures.MELSPECS)], data[use_features.index(AudioFeatures.MFCC)], data[
                                                 -1]
                    spec = spec.cuda()
                    mel = mel.cuda()
                    mfcc = mfcc.cuda()
                    label = label.cuda()

                    output = model(spec, mel, mfcc)
                    _, predicted = torch.max(func.softmax(output, dim=1), 1)
                    total += label.size(0)
                    correct += (predicted == label).sum().item()
                    acc = correct / total
                    bar_test.set_postfix(acc=acc)
                    bar_test.update(1)
            final = correct / total
            bar_test.close()
            now_time = time.time()
            logger.info(
                "Finish testing general model, for fold {}/{}, epoch {}, time cost {}s ,with acc {}".format(
                    current_fold,
                    total_fold,
                    current_epoch,
                    now_time - current_time,
                    final))
            save_name = os.path.join(save_dir, "fold{}_{}-epoch{}-loss{}-acc{}.pth").format(current_fold,
                                                                                            total_fold,
                                                                                            current_epoch, losses,
                                                                                            final)
            torch.save(model.state_dict(), save_name)
            logger.info("Saving weight to {}".format(save_name))
