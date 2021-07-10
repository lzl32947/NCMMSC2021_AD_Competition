import json
import os.path

from torch.utils.data.dataloader import DataLoader

from configs.types import AudioFeatures
from util.data_loader import AldsDataset
from util.files_util import set_working_dir, read_config
import time

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
        print("mfcc->{}\tspec->{}\tmelspec->{}\tlabel:{}\ttime use:{:<.2f}s".format(item[0].shape, item[1].shape,
                                                                                    item[2].shape, item[3],
                                                                                    current_time - now_time))
        now_time = current_time
        print(item[0].numpy().mean(), item[0].numpy().std())
