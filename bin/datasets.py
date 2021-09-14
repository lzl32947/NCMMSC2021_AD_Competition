import json
import os

import librosa
import numpy as np
from torch.utils.data.dataloader import DataLoader

from configs.types import AudioFeatures, DatasetMode
from util.train_util.data_loader import AldsTorchDataset, AldsDataset
from util.tools.files_util import global_init
import time
import matplotlib.pyplot as plt

from util.log_util.logger import GlobalLogger
from util.train_util.trainer_util import prepare_feature, prepare_dataloader

if __name__ == '__main__':
    """
    This file is used to test the datasets(dataloader), and be stored in directory 'bin'
    """
    # Get the global identifier and the configs
    time_identifier, configs = global_init()
    # Get the logger
    logger = GlobalLogger().get_logger()
    # Whether to show the images
    show_img = True
    # Init the features to use
    use_features = prepare_feature(configs['features'])
    # use_features = [AudioFeatures.MFCC]
    dataset_mode = DatasetMode.EVAL5
    # Save image to output
    save = False
    output_dir = os.path.join(configs["output"]["output_dir"], time_identifier)
    count = 0
    # Get the dataloader from the generator
    for dataloader in prepare_dataloader(use_features, configs["dataset"], dataset_mode):
        logger.info("Using config:" + json.dumps(configs['dataset']['process'], ensure_ascii=False))
        # Calculate the process time
        now_time = time.time()
        for item in dataloader:
            # Store the time of processing
            current_time = time.time()
            # Write log
            logging_str = ""
            for index, features in enumerate(use_features):
                logging_str = logging_str + ("{}->{}\t".format(use_features[index].value, item[features].shape))
            if dataset_mode != DatasetMode.EVAL5 and dataset_mode != DatasetMode.EVAL30:
                logging_str = logging_str + "label: {}\t".format(item[AudioFeatures.LABEL])
            logging_str = logging_str + "time use: {:<.2f}".format(current_time - now_time)
            logger.info(logging_str)
            if show_img:
                # Get the batch
                batch_size = len(item[AudioFeatures.NAME])
                for batch_num in range(batch_size):
                    # Plot the data in each figure
                    fig = plt.figure()
                    plot_position = 1
                    for index, features in enumerate(item.keys()):
                        # Ignore the label
                        if features == AudioFeatures.LABEL:
                            continue
                        if features == AudioFeatures.NAME:
                            fig.suptitle(str(item[AudioFeatures.NAME][batch_num]))
                            continue
                        # Add the subplot to figure
                        if dataset_mode == DatasetMode.EVAL5 or dataset_mode == DatasetMode.EVAL30:
                            ax = fig.add_subplot(len(item.keys()) - 1, 1, plot_position)
                        else:
                            ax = fig.add_subplot(len(item.keys()) - 2, 1, plot_position)
                        plot_position += 1

                        # If in format of Mat(2 dimension) then use the matshow()
                        if len(item[features][batch_num].shape) == 2:
                            ax.matshow(item[features][batch_num], aspect='auto')
                        # In format of Image(3 dimension) and use the imshow()
                        elif len(item[features][batch_num].shape) == 3:
                            img = np.transpose(item[features][batch_num], (1, 2, 0))
                            ax.imshow(img, aspect='auto')
                        # In format of Audio(1 dimension) and use the plot()
                        elif len(item[features][batch_num].shape) == 1:
                            ax.plot(range(len(item[features][batch_num])), item[features][batch_num])
                        # Add the title
                        ax.set_title("{}".format(list(item.keys())[index].value))
                    # Plot the image
                    plt.tight_layout()
                    if save:
                        fig.savefig(os.path.join(output_dir, "{}-{}.png".format(count, batch_num)))
                    else:
                        fig.show()
                    plt.close(fig)
            # Update the time
            now_time = time.time()
            if save:
                if count > 10:
                    break
                else:
                    count += 1
