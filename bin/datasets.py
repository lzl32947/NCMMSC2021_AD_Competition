import json
import os.path

from torch.utils.data.dataloader import DataLoader

from configs.types import AudioFeatures
from util.data_loader import AldsDataset
from util.files_util import set_working_dir, read_config
import time
import matplotlib.pyplot as plt

if __name__ == '__main__':
    set_working_dir("./..")
    configs = read_config(os.path.join("configs", "config.yaml"))

    use_features = []
    for item in AudioFeatures:
        if item.value in configs['features']:
            use_features.append(item)

    dataset = AldsDataset(use_features=use_features, use_merge=True,
                          crop_count=5, sample_length=15, sr=16000, configs=configs['process'])
    print("Using config:")
    print(json.dumps(configs['process'], indent=1, separators=(', ', ': '), ensure_ascii=False))
    dataloader = DataLoader(dataset, batch_size=1)
    now_time = time.time()
    for item in dataloader:
        current_time = time.time()
        for index, features in enumerate(use_features):
            print("{}->{}".format(use_features[index].value, item[index].shape), end="\t")
        print("label: {}".format(item[-1]), end="\t")
        print("time use: {:<.2f}".format(current_time - now_time))

        batch_size = item[0].shape[0]
        for batch_num in range(batch_size):
            fig = plt.figure()
            plot_position = 1
            for index, features in enumerate(use_features):
                ax = fig.add_subplot(len(use_features), 1, plot_position)
                plot_position += 1

                ax.matshow(item[index][batch_num])
                ax.set_title("{}".format(use_features[index].value))
            plt.tight_layout()
            fig.show()
            plt.close(fig)

        now_time = time.time()
