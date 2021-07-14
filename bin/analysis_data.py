import os
import numpy as np
import matplotlib.pyplot as plt
import contextlib
import wave

from configs.types import ADType
from util.tools.files_util import set_working_dir


def plot_distribution(data_dir: str):
    """

    :param data_dir:
    :return:
    """
    return_tuple = []
    for t in ADType:
        item = t.value
        sub_dir = os.path.join(data_dir, item)
        for audios in os.listdir(sub_dir):
            split_tuple = audios.replace("-", "_").replace(".wav", "").split("_")
            return_tuple.append(split_tuple)
    return return_tuple


def plot_males(input_tuple, total=False):
    map_dict = {}
    v = 0
    for t in ADType:
        item = t.value
        map_dict[item] = v
        v += 1
    F_list = [0, 0, 0]
    M_list = [0, 0, 0]
    T_list = [0, 0, 0]
    for item in input_tuple:
        if item[1] == "F":
            F_list[map_dict[item[0]]] += 1
        else:
            M_list[map_dict[item[0]]] += 1
        T_list[map_dict[item[0]]] += 1
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)

    x_label = [t.value for t in ADType]
    x_ticks = np.arange(len(x_label))
    if total:
        width = 0.2
        rects1 = ax.bar(x_ticks - width / 2, M_list, width, label='Male', fc="blue")
        rects2 = ax.bar(x_ticks + width / 2, F_list, width, label='Female', fc="red")
        rects3 = ax.bar(x_ticks + 3 * width / 2, T_list, width, label='Total', fc="green")
        ax.bar_label(rects1, padding=2)
        ax.bar_label(rects2, padding=2)
        ax.bar_label(rects3, padding=3)
        ax.set_xticks(x_ticks + width / 2)
        ax.set_xticklabels(x_label)
    else:
        width = 0.4
        rects1 = ax.bar(x_ticks - width / 2, M_list, width, label='Male', fc="blue")
        rects2 = ax.bar(x_ticks + width / 2, F_list, width, label='Female', fc="red")
        ax.bar_label(rects1, padding=2)
        ax.bar_label(rects2, padding=2)
        ax.set_xticks(x_ticks)
        ax.set_xticklabels(x_label)
    ax.set_ylabel('Samples')
    ax.set_title('Samples ordered by type')
    ax.legend()

    fig.tight_layout()

    fig.show()
    plt.close(fig)


def plot_length(input_tuple, data_dir, use_merge=True):
    for t in ADType:
        item = t.value
        F_list = []
        M_list = []
        T_list = []
        for audio_name in input_tuple:
            if audio_name[0] == item:
                if not use_merge:
                    file_name = os.path.join(data_dir, item,
                                             "{}_{}_{}_{}.wav".format(audio_name[0], audio_name[1], audio_name[2],
                                                                      audio_name[3]))
                else:
                    file_name = os.path.join(data_dir, item,
                                             "{}_{}_{}.wav".format(audio_name[0], audio_name[1], audio_name[2],
                                                                   ))
                with contextlib.closing(wave.open(file_name, 'r')) as f:
                    frames = f.getnframes()
                    rate = f.getframerate()
                    wav_length = frames / float(rate)

                if audio_name[1] == "F":
                    F_list.append(wav_length)
                else:
                    M_list.append(wav_length)
                T_list.append(wav_length)
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        ax.hist(M_list, bins=10, density=False, color="blue")
        ax.set_title("Length of audio of {} with {}".format(item, "Male"))
        fig.show()
        plt.close(fig)
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        ax.hist(F_list, bins=10, density=False, color="red")
        ax.set_title("Length of audio of {} with {}".format(item, "Female"))
        fig.show()
        plt.close(fig)
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        ax.hist(T_list, bins=10, density=False, color="green")
        ax.set_title("Length of audio of {}".format(item))
        fig.show()
        plt.close(fig)


if __name__ == '__main__':
    # when loading this files, the working dir should be set to the upper directory.
    set_working_dir("./..")

    adjusted = plot_distribution("dataset/merge")

    plot_males(adjusted, total=True)
    plot_length(adjusted, data_dir="dataset/merge")
