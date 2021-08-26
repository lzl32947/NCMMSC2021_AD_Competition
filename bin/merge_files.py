import os
import soundfile
import numpy as np

from configs.types import ADType
from util.tools.files_util import create_dir, global_init
from tqdm import tqdm

if __name__ == '__main__':
    """
    This file is used to merge all the audios that from the same person to the same audio file
    """
    # Default directory are 'dataset/raw' and 'dataset/merge'
    base_dir = os.path.join("dataset", "raw")
    target_dir = os.path.join("dataset", "merge")

    # Read config and init the global identifier
    time_identifier, config = global_init("config")

    if not create_dir(target_dir):
        raise RuntimeError("Target dir not created!")

    # Traverse the ADTypes to merge by the types
    for t in ADType:
        types = t.value
        # Generate the sub-directory
        # Target sub-directory
        target_sub_dir = os.path.join(target_dir, types)
        # Corresponding source sub-directory
        base_sub_dir = os.path.join(base_dir, types)
        # If not exist the directory then create
        create_dir(target_sub_dir)
        # Get all the files in the source directory
        src_files = os.listdir(base_sub_dir)
        # Generate the target dict to contain the audio from the same person
        target_dict = {}
        # item in the format of 'Type-(Fe)male-identifier-slip.wav'
        for item in src_files:
            identifier = item[:len(types) + 9]
            if identifier not in target_dict.keys():
                target_dict[identifier] = [item, ]
            else:
                target_dict[identifier].append(item)
        length = len(target_dict.keys())
        # Create the tqdm bar for visual aid
        bar = tqdm(range(length))
        bar.set_description("Combining files in {}".format(types))
        # For each person in the dict.keys(), combine all the audios
        for item in target_dict.keys():
            item_list = target_dict[item]
            # Sort the list to merge them in the correct order
            item_list.sort()
            target_file = os.path.join(target_sub_dir, "{}.wav".format(item))
            # The audio wav stores in the combine_x
            combine_x = []
            # The audio sample rate stores in the combine_sr
            combine_sr = []
            for radios in item_list:
                # Read the file and then merge them into one file
                x, sr = soundfile.read(os.path.join(base_sub_dir, radios))
                combine_x.append(x)
                combine_sr.append(sr)
            # Stack them with numpy
            combine_x = np.hstack(combine_x)
            # NOTICE: the sample rate should be same and here only use the first sample rate as the total
            soundfile.write(target_file, combine_x, combine_sr[0])
            # Update the bar
            bar.update(1)
        bar.close()
