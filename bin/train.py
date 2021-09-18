import os
import time
from typing import Dict, Optional, List
import torch.nn.functional as func
from torch import optim, nn
from tqdm import tqdm
from configs.types import AudioFeatures, DatasetMode
from model.manager import Registers
from util.log_util.logger import GlobalLogger
from util.tools.files_util import global_init, create_dir
import torch

from util.train_util.trainer_util import prepare_dataloader, get_best_loss_weight


def train_joint(configs: Dict, time_identifier: str, model_name: str, base_model_name: str,
                use_features: List[AudioFeatures], train_specific: bool = True,
                train_specific_epoch: int = 20, train_general_epoch: int = 40, specific_weight: Optional[Dict] = None,
                general_weight: Optional[str] = None, train_general: bool = False,
                fine_tune: bool = True, fine_tune_epoch: int = 20, input_channels: int = 1, **kwargs) -> None:
    """
    This is the trainer of training with joint-features.
    :param input_channels: int, the input channels of the model, default is 1
    :param base_model_name: str, the name of base model (extraction model)
    :param fine_tune_epoch: int, the epochs if fine-tune the general model
    :param train_general: bool, whether to train the general model or directly use the given weight
    :param general_weight: str, if not train the general model, the weights must be given
    :param specific_weight: Dict, if not train the specific model, the weights must be given
    :param train_general_epoch: int, the epochs if train the general model
    :param train_specific_epoch: int, the epochs if train the specific model
    :param train_specific: bool, whether to train the specific model or directly use the given weight
    :param model_name: str, name of the used model
    :param configs: Dict, the configs
    :param time_identifier: str, the global identifier
    :param fine_tune: bool, whether to fine-tune the model
    :return: None
    """
    # Init the logger
    logger = GlobalLogger().get_logger()
    logger.info("Training the general model for competition.")
    # Train the specific model, this usually happens when there is no previous training

    if train_specific:
        logger.info("Training the specific model with {}.".format(base_model_name))
        for specific_feature in use_features:
            logger.info(
                "Training the specific model with feature {} in {} epochs.".format(specific_feature.name,
                                                                                   train_specific_epoch))
            # Init the weight saving directory
            save_dir = os.path.join(configs['weight']['weight_dir'], time_identifier, specific_feature.value)
            create_dir(save_dir)

            for train_dataloader in prepare_dataloader(use_features=[specific_feature], configs=configs["dataset"],
                                                       run_for=DatasetMode.TRAIN):

                # If not running on GPU
                model = Registers.model[base_model_name]()
                model = model.cuda()
                # Init the criterion, CE by default
                criterion = nn.CrossEntropyLoss()
                # Init the optimizer, SGD by default
                optimizer = optim.AdamW(model.parameters(), lr=2e-4)
                scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=2e-4, epochs=train_specific_epoch,
                                                                steps_per_epoch=len(train_dataloader),
                                                                anneal_strategy="linear")
                for current_epoch in range(1, train_specific_epoch + 1):
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
                        "Training using feature {}, epoch {}".format(specific_feature,
                                                                     current_epoch))
                    # Running one batch
                    for iteration, data in enumerate(train_dataloader):
                        feature, label = data[specific_feature], data[AudioFeatures.LABEL]
                        if input_channels != 1:
                            feature = torch.cat([feature] * input_channels, dim=1)
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
                        # Update the scheduler
                        scheduler.step()
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
                        "Finish training feature {}, epoch {}, time cost {:.2f}s ,with loss {:.5f}".format(
                            specific_feature,

                            current_epoch,
                            now_time - current_time,
                            losses))
                    # Re-init the timer
                    current_time = time.time()

                    # Time the timer
                    now_time = time.time()
                    # Save the weight to the directory
                    save_name = os.path.join(save_dir, "epoch{}-loss{}.pth").format(
                        current_epoch,
                        losses)
                    torch.save(model.state_dict(), save_name)
                    # Write the log
                    logger.info("Saving weight to {}".format(save_name))
                logger.info("Finish training for feature {}.".format(specific_feature.name))
        logger.info("Finishing training for all features.")

    else:
        logger.info("Skip training the specific features.")

    if train_general:
        # Init the saving directory
        save_dir = os.path.join(configs['weight']['weight_dir'], time_identifier, "General")
        create_dir(save_dir)
        logger.info("Training general models.")
        for train_dataloader in prepare_dataloader(use_features, configs["dataset"], DatasetMode.TRAIN):
            # Send the model to GPU
            model = Registers.model[model_name]()
            model.cuda()
            # Init the criterion, CE by default
            criterion = nn.CrossEntropyLoss()
            # Init the optimizer, SGD by default
            optimizer = optim.AdamW(model.parameters(), lr=2e-4)
            scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=2e-4, epochs=train_general_epoch,
                                                            steps_per_epoch=len(train_dataloader),
                                                            anneal_strategy="linear")
            # Load weight
            if train_specific:
                # Load weight from all separated directories
                for specific_feature in use_features:
                    # By default the bast accuracy weight is used
                    weight_file = get_best_loss_weight(os.path.join(configs['weight']['weight_dir'], time_identifier),
                                                       specific_feature)
                    # Load weights
                    if specific_feature == AudioFeatures.SPECS:
                        model.extractor_spec.load_state_dict(torch.load(weight_file), strict=False)
                    if specific_feature == AudioFeatures.MELSPECS:
                        model.extractor_mel.load_state_dict(torch.load(weight_file), strict=False)
                    if specific_feature == AudioFeatures.MFCC:
                        model.extractor_mfcc.load_state_dict(torch.load(weight_file), strict=False)
                    # Write the logs
                    logger.info("Load weight {} for {}.".format(weight_file, specific_feature.value))
            else:
                assert specific_weight is not None
                assert AudioFeatures.MFCC in specific_weight.keys()
                assert AudioFeatures.MELSPECS in specific_weight.keys()
                assert AudioFeatures.SPECS in specific_weight.keys()
                # Load weight from all separated directories
                for specific_feature in use_features:
                    # By default the bast accuracy weight is used
                    weight_file = get_best_loss_weight(
                        os.path.join(configs['weight']['weight_dir'], specific_weight[specific_feature]),
                        specific_feature)
                    # Load weights
                    if specific_feature == AudioFeatures.SPECS:
                        model.extractor_spec.load_state_dict(torch.load(weight_file), strict=False)
                    if specific_feature == AudioFeatures.MELSPECS:
                        model.extractor_mel.load_state_dict(torch.load(weight_file), strict=False)
                    if specific_feature == AudioFeatures.MFCC:
                        model.extractor_mfcc.load_state_dict(torch.load(weight_file), strict=False)
                    # Write the logs
                    logger.info("Load weight {} for {}.".format(weight_file, specific_feature.value))
            # Unfrozen weights
            for params in model.extractor_spec.parameters():
                params.requires_grad = False
            for params in model.extractor_mfcc.parameters():
                params.requires_grad = False
            for params in model.extractor_mel.parameters():
                params.requires_grad = False
            logger.info("In mode training, freeze the extractor layers.")
            # Running epoch
            for current_epoch in range(1, train_general_epoch + 1):
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
                    "Training general model epoch {}".format(
                        current_epoch))
                # Running one batch
                for iteration, data in enumerate(train_dataloader):
                    # Get features and set them to cuda
                    spec, mel, mfcc, label = data[AudioFeatures.SPECS], data[AudioFeatures.MELSPECS], data[
                        AudioFeatures.MFCC], data[AudioFeatures.LABEL]
                    if input_channels != 1:
                        spec = torch.cat([spec] * input_channels, dim=1)
                        mel = torch.cat([mel] * input_channels, dim=1)
                        mfcc = torch.cat([mfcc] * input_channels, dim=1)
                    spec = spec.cuda()
                    mel = mel.cuda()
                    mfcc = mfcc.cuda()
                    label = label.cuda()
                    # Set the optimizer to zero
                    optimizer.zero_grad()
                    # Go through one epoch
                    output = model(mfcc, spec, mel)
                    # Calculate the loss
                    loss = criterion(output, label)
                    # Back propagation
                    loss.backward()
                    # Update the optimizer
                    optimizer.step()
                    # Update the scheduler
                    scheduler.step()
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
                    "Finish training general model, epoch {}, time cost {:.2f}s ,with loss {:.5f}".format(

                        current_epoch,
                        now_time - current_time,
                        losses))
                # Re-init the timer
                current_time = time.time()
                # Save the weight to the directory
                save_name = os.path.join(save_dir, "epoch{}-loss{}.pth").format(
                    current_epoch, losses,
                )
                torch.save(model.state_dict(), save_name)
                # Write the log
                logger.info("Saving weight to {}".format(save_name))
    else:
        logger.info("Skip training the general model.")
    if fine_tune:
        # Init the saving directory
        save_dir = os.path.join(configs['weight']['weight_dir'], time_identifier, "Fine_tune")
        create_dir(save_dir)
        logger.info("Fine-tune general models.")

        for train_dataloader in prepare_dataloader(use_features, configs["dataset"], DatasetMode.TRAIN):

            # Send the model to GPU
            model = Registers.model[model_name]()
            model.cuda()

            # Init the criterion, CE by default
            criterion = nn.CrossEntropyLoss()
            # Init the optimizer, SGD by default
            optimizer = optim.AdamW(model.parameters(), lr=2e-4)
            scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=2e-4, epochs=train_general_epoch,
                                                            steps_per_epoch=len(train_dataloader),
                                                            anneal_strategy="linear")

            if train_general:

                # By default the bast accuracy weight is used
                weight_file = get_best_loss_weight(os.path.join(configs['weight']['weight_dir'], time_identifier),
                                                   "General")
                # Load weights
                model.load_state_dict(torch.load(weight_file), strict=True)
                # Write the logs
                logger.info("Load weight {} for fine-tuning.".format(weight_file))
            else:
                assert general_weight is not None
                # By default the bast accuracy weight is used
                weight_file = get_best_loss_weight(os.path.join(configs['weight']['weight_dir'], general_weight),
                                                   "General")
                # Load weights
                model.load_state_dict(torch.load(weight_file), strict=True)
                # Write the logs
                logger.info("Load weight {} for fine-tuning.".format(weight_file))

            # If fine-tune, the parameters in extractor should be unfreeze
            for params in model.extractor_spec.parameters():
                params.requires_grad = True
            for params in model.extractor_mfcc.parameters():
                params.requires_grad = True
            for params in model.extractor_mel.parameters():
                params.requires_grad = True

            logger.info("In mode fine-tune, unfreeze the extractor layers.")
            # Running epoch
            for current_epoch in range(1, fine_tune_epoch + 1):
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
                    "Fine-tuning general model, epoch {}".format(
                        current_epoch))
                # Running one batch
                for iteration, data in enumerate(train_dataloader):
                    # Get features and set them to cuda
                    spec, mel, mfcc, label = data[AudioFeatures.SPECS], data[AudioFeatures.MELSPECS], data[
                        AudioFeatures.MFCC], data[AudioFeatures.LABEL]

                    if input_channels != 1:
                        spec = torch.cat([spec] * input_channels, dim=1)
                        mel = torch.cat([mel] * input_channels, dim=1)
                        mfcc = torch.cat([mfcc] * input_channels, dim=1)

                    spec = spec.cuda()
                    mel = mel.cuda()
                    mfcc = mfcc.cuda()
                    label = label.cuda()

                    # Set the optimizer to zero
                    optimizer.zero_grad()
                    # Go through one epoch
                    output = model(mfcc, spec, mel)
                    # Calculate the loss
                    loss = criterion(output, label)
                    # Back propagation
                    loss.backward()
                    # Update the optimizer
                    optimizer.step()
                    # Update the scheduler
                    scheduler.step()
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
                    "Finish fine-tune general model, epoch {}, time cost {:.2f}s ,with loss {:.5f}".format(

                        current_epoch,
                        now_time - current_time,
                        losses))
                # Re-init the timer
                current_time = time.time()

                # Save the weight to the directory
                save_name = os.path.join(save_dir, "epoch{}-loss{}.pth").format(
                    current_epoch, losses
                )
                torch.save(model.state_dict(), save_name)
                # Write the log
                logger.info("Saving weight to {}".format(save_name))
            logger.info("Finish fine-tuning the model.")
    else:
        logger.info("Skip fine-tune the model.")


if __name__ == '__main__':
    """
    This is a template for joint-features training
    """
    # Init the global environment
    time_identifier, configs = global_init(for_competition=True)
    logger = GlobalLogger().get_logger()
    # Train the general model
    model_name = "CompetitionSpecificTrainVggNet19BNBackboneLongModel"
    base_model_name = "CompetitionSpecificTrainVggNet19BNBackboneLongModel"
    logger.info("Training with model {}.".format(model_name))
    train_joint(configs, time_identifier, model_name, base_model_name,
                use_features=[AudioFeatures.MFCC, AudioFeatures.SPECS, AudioFeatures.MELSPECS],
                train_specific=True, train_specific_epoch=20,
                train_general=False, train_general_epoch=20,
                fine_tune=False, fine_tune_epoch=20, input_channels=3)
