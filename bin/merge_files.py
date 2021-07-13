import os
import soundfile
import numpy as np

from configs.types import ADType
from util.files_util import create_dir, set_working_dir, check_dir
from tqdm import tqdm

if __name__ == '__main__':
    base_dir = "dataset/raw"
    target_dir = "dataset/merge"

    # when loading this files, the working dir should be set to the upper directory.
    set_working_dir("./..")
    check_dir()

    if not create_dir(target_dir):
        raise RuntimeError("Target dir not created!")

    for t in ADType:
        types = t.value
        target_sub_dir = os.path.join(target_dir, types)
        base_sub_dir = os.path.join(base_dir, types)

        create_dir(target_sub_dir)
        src_files = os.listdir(base_sub_dir)

        target_dict = {}
        for item in src_files:
            identifier = item[:len(types) + 9]
            if identifier not in target_dict.keys():
                target_dict[identifier] = [item, ]
            else:
                target_dict[identifier].append(item)
        length = len(target_dict.keys())
        bar = tqdm(range(length))
        bar.set_description("Combining files in {}".format(types))
        for item in target_dict.keys():
            item_list = target_dict[item]
            item_list.sort()
            target_file = os.path.join(target_sub_dir, "{}.wav".format(item))
            combine_x = []
            combine_sr = []
            for radios in item_list:
                x, sr = soundfile.read(os.path.join(base_sub_dir, radios))
                combine_x.append(x)
                combine_sr.append(sr)
            combine_x = np.hstack(combine_x)
            soundfile.write(target_file, combine_x, combine_sr[0])
            bar.update(1)
        bar.close()
