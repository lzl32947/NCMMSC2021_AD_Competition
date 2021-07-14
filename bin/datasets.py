import json

import numpy as np
from torch.utils.data.dataloader import DataLoader

from configs.types import AudioFeatures, DatasetMode
from util.train_util.data_loader import AldsDataset
from util.tools.files_util import global_init
import time
import matplotlib.pyplot as plt

from util.log_util.logger import GlobalLogger
from util.train_util.trainer_util import prepare_feature, prepare_dataloader

if __name__ == '__main__':
    time_identifier, configs = global_init()
    logger = GlobalLogger().get_logger()
    use_features = prepare_feature(configs['features'])

    for dataloader in  prepare_dataloader(use_features, configs["dataset"], DatasetMode.TRAIN, k_fold=0):
        logger.info("Using config:" + json.dumps(configs['dataset']['process'], ensure_ascii=False))
        now_time = time.time()
        for item in dataloader:
            current_time = time.time()
            logging_str = ""
            for index, features in enumerate(use_features):
                logging_str = logging_str + ("{}->{}\t".format(use_features[index].value, item[index].shape))
            logging_str = logging_str + "label: {}\t".format(item[-1])
            logging_str = logging_str + "time use: {:<.2f}".format(current_time - now_time)
            logger.info(logging_str)

            batch_size = item[0].shape[0]
            for batch_num in range(batch_size):
                fig = plt.figure()
                plot_position = 1
                for index, features in enumerate(use_features):
                    ax = fig.add_subplot(len(use_features), 1, plot_position)
                    plot_position += 1
                    if len(item[index][batch_num].shape) == 2:
                        ax.matshow(item[index][batch_num])
                    else:
                        img = np.transpose(item[index][batch_num], (1, 2, 0))
                        ax.imshow(img)
                    ax.set_title("{}".format(use_features[index].value))
                plt.tight_layout()
                fig.show()
                plt.close(fig)

            now_time = time.time()
