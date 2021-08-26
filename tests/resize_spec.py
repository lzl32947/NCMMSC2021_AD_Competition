from configs.types import AudioFeatures, DatasetMode
from network.spec.spec import SpecModel
import json
import os.path
from torch import nn
from torch import optim
from torch.utils.data.dataloader import DataLoader
import torch
from util.train_util.data_loader import AldsDataset
from util.tools.files_util import global_init
from tqdm import tqdm
import torch.nn.functional as func

from util.log_util.logger import GlobalLogger

if __name__ == '__main__':
    time_identifier, configs = global_init("config")
    logger = GlobalLogger().get_logger()
    use_features = []
    for item in AudioFeatures:
        if item.value in configs['features']:
            use_features.append(item)
    k_fold = 5

    logger.info("Using config:" + json.dumps(configs['process'], ensure_ascii=False))
    acc_list = [[] for i in range(k_fold)]
    for current_fold in range(k_fold):
        model = SpecModel()
        model.cuda()

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

        train_dataset = AldsDataset(use_features=use_features, use_merge=True,
                                    repeat_times=32, configs=configs['process'], k_fold=k_fold,
                                    current_fold=current_fold, random_disruption=True,
                                    run_for=DatasetMode.TRAIN)

        train_dataloader = DataLoader(train_dataset, batch_size=16)

        test_dataset = AldsDataset(use_features=use_features, use_merge=True,
                                   repeat_times=32, configs=configs['process'], k_fold=k_fold,
                                   current_fold=current_fold, random_disruption=True,
                                   run_for=DatasetMode.TEST)

        test_dataloader = DataLoader(test_dataset, batch_size=16)

        for epoch in range(10):
            model.train()
            length = len(train_dataloader)
            running_loss = 0.0
            bar = tqdm(range(length))
            bar.set_description("Training for epoch {}".format(epoch))
            for iteration, data in enumerate(train_dataloader):
                spec, label = data[use_features.index(AudioFeatures.MELSPECS)], data[-1]

                spec = spec.cuda()
                label = label.cuda()


                optimizer.zero_grad()

                output = model(spec)

                loss = criterion(output, label)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()

                bar.set_postfix(loss=running_loss / (iteration + 1))
                bar.update(1)

            bar.close()

            correct = 0
            total = 0
            length = len(test_dataloader)
            bar_test = tqdm(range(length))
            bar_test.set_description("Testing for epoch {}".format(epoch))
            model.eval()
            with torch.no_grad():
                for data in test_dataloader:
                    spec, label = data[use_features.index(AudioFeatures.MELSPECS)], data[-1]
                    spec = spec.cuda()
                    label = label.cuda()
                    outputs = model(spec)
                    _, predicted = torch.max(func.softmax(outputs,dim=1), 1)
                    total += label.size(0)
                    correct += (predicted == label).sum().item()
                    acc = correct / total
                    bar_test.set_postfix(acc=acc)
                    bar_test.update(1)
            final = correct / total
            acc_list[current_fold].append(final)
            bar_test.close()

            torch.save(model.state_dict(), os.path.join("weight", "{}-{}-{}-spec.pth".format(k_fold, current_fold, epoch)))
    logger.info(acc_list)
