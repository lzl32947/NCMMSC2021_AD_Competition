from torch.utils.data.dataloader import DataLoader

from configs.types import OutputType
from util.data_loader import AldsDataset
from util.files_util import set_working_dir

if __name__ == '__main__':
    set_working_dir("./..")
    dataset = AldsDataset(output_type=[OutputType.MFCC, OutputType.SPECS, OutputType.MELSPECS], use_merge=True,
                          crop_count=5)
    dataloader = DataLoader(dataset)
    for item in dataloader:
        print("mfcc->{}\tspec->{}\tmelspec->{}\tlabel:{}".format(item[0].shape, item[1].shape, item[2].shape, item[3]))
