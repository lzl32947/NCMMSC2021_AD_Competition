from configs.types import AudioFeatures, DatasetMode
from network.melspec.melspec import MelSpecModel
import json
import os.path
from torch import nn
from torch import optim
from torch.utils.data.dataloader import DataLoader
import torch
from util.data_loader import AldsDataset
from util.files_util import set_working_dir, read_config
from tqdm import tqdm

if __name__ == '__main__':
    set_working_dir("./..")
    configs = read_config(os.path.join("configs", "config.yaml"))

    use_features = []
    for item in AudioFeatures:
        if item.value in configs['features']:
            use_features.append(item)
    k_fold = 5

    print("Using config:")
    print(json.dumps(configs['process'], indent=1, separators=(', ', ': '), ensure_ascii=False))
    acc_list = [[] for i in range(k_fold)]
    for current_fold in range(k_fold):
        model = MelSpecModel()
        model.cuda()
        model.train()

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
            length = len(train_dataloader)
            running_loss = 0.0
            bar = tqdm(range(length))
            bar.set_description("Training for epoch {}".format(epoch))
            for iteration, data in enumerate(train_dataloader):
                melspec, label = data[use_features.index(AudioFeatures.MELSPECS)], data[-1]
                melspec = melspec.cuda()
                label = label.cuda()
                output = model(melspec)
                loss = criterion(output, label)
                optimizer.zero_grad()
                bar.set_postfix(loss=loss.item())
                bar.update(1)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
            bar.close()

            correct = 0
            total = 0
            length = len(test_dataloader)
            bar_test = tqdm(range(length))
            bar_test.set_description("Testing for epoch {}".format(epoch))
            with torch.no_grad():
                for data in test_dataloader:
                    melspec, label = data[use_features.index(AudioFeatures.MELSPECS)], data[-1]
                    melspec = melspec.cuda()
                    label = label.cuda()
                    outputs = model(melspec)
                    _, predicted = torch.max(outputs.data, 1)
                    total += label.size(0)
                    correct += (predicted == label).sum().item()
                    acc = correct / total
                    bar_test.set_postfix(acc=acc)
                    bar_test.update(1)
            final = correct / total
            acc_list[current_fold].append(final)
            bar_test.close()
    print(acc_list)