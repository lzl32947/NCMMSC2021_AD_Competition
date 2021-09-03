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
from util.train_util.trainer_util import prepare_feature, prepare_dataloader, read_weight, get_best_acc_weight
import torch


def train_joint(configs: Dict, time_identifier: str, model_name: str, use_features: List[AudioFeatures],
                fine_tune: bool = False, load_weight_identifier: Optional[str] = None,
                weighted_dir: Optional[str] = None) -> None:
    """
    This is the trainer of training with joint-features.
    :param model_name: str, name of the used model
    :param configs: Dict, the configs
    :param time_identifier: str, the global identifier
    :param use_features: List[AudioFeatures], the feature to use
    :param fine_tune: bool, whether to fine-tune the model
    :param load_weight_identifier: str, if given, the weight will load from the given directory instead of the time_identifier
    :param weighted_dir: str, the name of the directory to load weight from
    :return: None
    """
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
            zip(prepare_dataloader(use_features, configs["dataset"], DatasetMode.TRAIN),
                prepare_dataloader(use_features, configs["dataset"], DatasetMode.TEST))):
        # Getting the epoch
        epoch = configs['train']['epoch']
        # Send the model to GPU
        model = Registers.model[model_name](input_shape=(128,157))
        model.cuda()

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
                spec, mel, mfcc, label = data[AudioFeatures.SPECS], data[AudioFeatures.MELSPECS], data[
                    AudioFeatures.MFCC], data[AudioFeatures.LABEL]
                spec = spec.cuda()
                mel = mel.cuda()
                mfcc = mfcc.cuda()
                label = label.cuda()
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
                    spec, mel, mfcc, label = data[AudioFeatures.SPECS], data[AudioFeatures.MELSPECS], data[
                        AudioFeatures.MFCC], data[AudioFeatures.LABEL]
                    spec = spec.cuda()
                    mel = mel.cuda()
                    mfcc = mfcc.cuda()
                    label = label.cuda()
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
    train_joint(configs, time_identifier, "MSMJointConcatFineTuneModel", use_features, True, "20210716_193130")
