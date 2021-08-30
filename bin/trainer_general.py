import logging
import time
from typing import Dict, List, Optional
from torch.utils.data.dataloader import DataLoader
from configs.types import AudioFeatures, DatasetMode
from network.general.general_model import GeneralModel, ConvModel
import os.path
from torch import nn
from torch import optim
import torch
from util.tools.files_util import global_init, create_dir
from tqdm import tqdm
import torch.nn.functional as func

from util.log_util.logger import GlobalLogger
from util.train_util.trainer_util import prepare_feature, prepare_dataloader, read_weight, get_best_acc_weight, \
    train_general

if __name__ == '__main__':
    """
    This is a template for joint-features training
    """
    # Init the global environment
    time_identifier, configs = global_init("config")
    logger = GlobalLogger().get_logger()
    use_features = prepare_feature(configs['features'])
    # Read the fold from config
    total_fold = configs['dataset']['k_fold']
    # Train the general model
    train_general(configs, time_identifier, use_features, False, "20210716_193130", cuda=1)
