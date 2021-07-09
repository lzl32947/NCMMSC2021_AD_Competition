from torch.utils.data.dataloader import DataLoader

from util.data_loader import AldsDataset
from util.files_util import set_working_dir

if __name__ == '__main__':
    set_working_dir("./..")
    dataset = AldsDataset(True, 5)
    dataloader = DataLoader(dataset)
    for item in dataloader:
        print(item)
