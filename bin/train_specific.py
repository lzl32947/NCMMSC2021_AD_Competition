import os
import time
from typing import Dict
import torch.nn.functional as func
from torch import optim, nn
from tqdm import tqdm
from configs.types import AudioFeatures, DatasetMode
from model.manager import Registers
from util.log_util.logger import GlobalLogger
from util.tools.files_util import global_init, create_dir
from util.train_util.trainer_util import prepare_feature, prepare_dataloader, read_weight, get_best_acc_weight
import torch


def train_specific_feature(configs: Dict, time_identifier: str, specific_feature: AudioFeatures,
                           model_func: callable, epoch: int = 20, **kwargs) -> None:
    """
    This is the trainer of training only with one specific features.
    :param epoch: int, training epochs
    :param configs: Dict, the configs
    :param time_identifier: str, the global identifier
    :param specific_feature: AudioFeatures, the feature to use
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
            zip(prepare_dataloader([specific_feature], configs["dataset"], DatasetMode.TRAIN),
                prepare_dataloader([specific_feature], configs["dataset"], DatasetMode.TEST))):

        # If not running on GPU
        model = model_func(**kwargs)
        model = model.cuda()

        # Init the criterion, CE by default
        criterion = nn.CrossEntropyLoss()
        # Init the optimizer, SGD by default
        optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

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
                feature, label = data[specific_feature], data[AudioFeatures.LABEL]
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
                    feature, label = data[specific_feature], data[AudioFeatures.LABEL]
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


if __name__ == '__main__':
    """
    This is a template for joint-features training
    """
    # Init the global environment
    time_identifier, configs = global_init()
    logger = GlobalLogger().get_logger()
    use_features = prepare_feature(configs['features'])
    # Read the fold from config
    total_fold = configs['dataset']['k_fold']
    # Train the general model
    model_func = Registers.model["SpecificTrainLongModel"]
    logger.info("Training with model {}.".format(model_func))
    train_specific_feature(configs, time_identifier, AudioFeatures.MELSPECS_VAD,
                           model_func,
                           input_shape=(1, 128, 782))