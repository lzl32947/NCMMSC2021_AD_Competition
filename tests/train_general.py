from configs.types import AudioFeatures, DatasetMode
from network.general.general_model import GeneralModel
import os.path
from torch import nn
from torch import optim
import torch
from util.tools.files_util import global_init
from tqdm import tqdm
import torch.nn.functional as func

from util.log_util.logger import GlobalLogger
from util.train_util.trainer_util import prepare_feature, prepare_dataloader

if __name__ == '__main__':
    time_identifier, configs = global_init("config")
    logger = GlobalLogger().get_logger()
    use_features = prepare_feature(configs['features'])

    # logger.info("Using config:" + json.dumps(configs['process'], ensure_ascii=False))

    acc_list = []
    for current_fold, (train_dataloader, test_dataloader) in enumerate(
            zip(prepare_dataloader(use_features,AldsDataset2D, configs["dataset"], DatasetMode.TRAIN),
                prepare_dataloader(use_features,AldsDataset2D, configs["dataset"], DatasetMode.TEST))):

        model = GeneralModel()
        model.cuda()

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

        for epoch in range(10):
            model.train()
            length = len(train_dataloader)
            running_loss = 0.0
            bar = tqdm(range(length))
            bar.set_description("Training for epoch {}".format(epoch))
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

            correct = 0
            total = 0
            length = len(test_dataloader)
            bar_test = tqdm(range(length))
            bar_test.set_description("Testing for epoch {}".format(epoch))
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
            acc_list[current_fold].append(final)
            bar_test.close()

            torch.save(model.state_dict(),
                       os.path.join("weight", "{}-{}-{}-spec.pth".format(current_fold, current_fold, epoch)))
    logger.info(acc_list)
