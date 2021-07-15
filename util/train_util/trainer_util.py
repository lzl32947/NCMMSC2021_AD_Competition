from typing import List, Dict, Generic
from torch.utils.data.dataloader import DataLoader
from configs.types import AudioFeatures, DatasetMode
from util.train_util.data_loader import AldsDataset


def prepare_feature(feature_list: List[str]) -> List[AudioFeatures]:
    """
    This function is to get which features are used in dataset.
    :param feature_list: List[str], and should be the configs['features']
    :return: list[AudioFeatures]
    """
    use_features = []
    # Traverse the feature list to convert them into AudioFeatures
    for item in AudioFeatures:
        if item.value in feature_list:
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
                                  run_for=DatasetMode.TRAIN)

            dataloader = DataLoader(dataset, batch_size=batch_size)
            yield dataloader
    else:
        # Generate the single dataloader
        for fold in range(1):
            dataset = AldsDataset(use_features=use_features, use_merge=use_merge,
                                  repeat_times=repeat_times, configs=configs['process'],
                                  random_disruption=random_disruption,
                                  run_for=DatasetMode.TRAIN)

            dataloader = DataLoader(dataset, batch_size=batch_size)
            yield dataloader
