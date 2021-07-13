from configs.types import AudioFeatures, DatasetMode
from network.melspec.melspec import MelSpecModel
import json
import os.path
from torch import nn
from torch import optim
from torch.utils.data.dataloader import DataLoader
import torch
from util.data_loader import AldsDataset
from util.files_util import global_init
from tqdm import tqdm
import torch.nn.functional as func

from util.logger import GlobalLogger

if __name__ == '__main__':
    time_identifier, configs = global_init()
    logger = GlobalLogger().get_logger()
    use_features = []
    for item in AudioFeatures:
        if item.value in configs['features']:
            use_features.append(item)
    k_fold = 5

    logger.info("Using config:")
    logger.info(json.dumps(configs['process'], ensure_ascii=False))
    acc_list = [[] for i in range(k_fold)]
    for current_fold in range(k_fold):
        model = MelSpecModel()
        model.cuda()

        test_dataset = AldsDataset(use_features=use_features, use_merge=True,
                                   repeat_times=32, configs=configs['process'], k_fold=k_fold,
                                   current_fold=current_fold, random_disruption=True,
                                   run_for=DatasetMode.TEST)

        test_dataloader = DataLoader(test_dataset, batch_size=16)

        model_path = os.path.join("weight", "{}-{}-{}-spec.pth".format(k_fold, current_fold, 1))
        stat_dict = torch.load(model_path)
        model.load_state_dict(stat_dict)

        correct = 0
        total = 0
        length = len(test_dataloader)
        bar_test = tqdm(range(length))
        bar_test.set_description("Testing for k-fold {}".format(current_fold))
        model.eval()
        with torch.no_grad():
            for data in test_dataloader:
                melspec, label = data[use_features.index(AudioFeatures.MELSPECS)], data[-1]
                melspec = melspec.cuda()
                label = label.cuda()
                outputs = model(melspec)
                _, predicted = torch.max(func.softmax(outputs,dim=1), 1)
                total += label.size(0)
                correct += (predicted == label).sum().item()
                acc = correct / total
                bar_test.set_postfix(acc=acc)
                bar_test.update(1)
        final = correct / total
        acc_list[current_fold].append(final)
        bar_test.close()

    logger.info(acc_list)
