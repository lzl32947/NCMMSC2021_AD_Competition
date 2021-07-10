import os.path

from torch.utils.data.dataloader import DataLoader

from configs.types import OutputType
from util.data_loader import AldsDataset
from util.files_util import set_working_dir, read_config
import time

if __name__ == '__main__':
    set_working_dir("./..")
    configs = read_config(os.path.join("configs", "config.yaml"))
    dataset = AldsDataset(output_type=[OutputType.MFCC, OutputType.SPECS, OutputType.MELSPECS], use_merge=True,
                          crop_count=5, configs=configs['process'])
    print("Using config:")
    print(configs['process'])
    dataloader = DataLoader(dataset)
    now_time = time.time()
    for item in dataloader:
        current_time = time.time()
        print("mfcc->{}\tspec->{}\tmelspec->{}\tlabel:{}\ttime use:{:<.2f}s".format(item[0].shape, item[1].shape,
                                                                                    item[2].shape, item[3],
                                                                                    current_time - now_time))
        now_time = current_time
