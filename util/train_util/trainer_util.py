from typing import List, Dict
from torch.utils.data.dataloader import DataLoader
from configs.types import AudioFeatures, DatasetMode
from util.train_util.data_loader import AldsDataset


def prepare_feature(feature_list: List[str]):
    use_features = []
    for item in AudioFeatures:
        if item.value in feature_list:
            use_features.append(item)
    return use_features


def prepare_dataloader(use_features: List[AudioFeatures], configs: Dict, run_for: DatasetMode, **kwargs):
    use_merge = configs['use_merge'] if 'use_merge' not in kwargs.keys() else kwargs['use_merge']
    repeat_times = configs['repeat_times'] if 'repeat_times' not in kwargs.keys() else kwargs['repeat_times']
    k_fold = configs['k_fold'] if 'k_fold' not in kwargs.keys() else kwargs['k_fold']
    batch_size = configs['batch_size'] if 'batch_size' not in kwargs.keys() else kwargs['batch_size']
    random_disruption = configs['random_disruption'] if 'random_disruption' not in kwargs.keys() else kwargs[
        'random_disruption']
    if k_fold != 0:
        for fold in range(k_fold):
            dataset = AldsDataset(use_features=use_features, use_merge=use_merge,
                                  repeat_times=repeat_times, configs=configs['process'], k_fold=k_fold,
                                  current_fold=fold, random_disruption=random_disruption,
                                  run_for=DatasetMode.TRAIN)

            dataloader = DataLoader(dataset, batch_size=batch_size)
            yield dataloader
    else:
        for fold in range(1):
            dataset = AldsDataset(use_features=use_features, use_merge=use_merge,
                                  repeat_times=repeat_times, configs=configs['process'],
                                  random_disruption=random_disruption,
                                  run_for=DatasetMode.TRAIN)

            dataloader = DataLoader(dataset, batch_size=batch_size)
            yield dataloader
