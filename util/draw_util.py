from typing import Union, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import torch


def draw_sound(wav: Union[torch.Tensor, np.ndarray, List], use_second: bool = True,
               sep: Optional[Union[torch.Tensor, np.ndarray, List]] = None, rate: Optional[int] = None) -> None:
    """
    Draw the wave of the sounds.
    :param wav: the wav files, should be in format of torch.Tensor, np.ndarray and list.
    :param sep: the separation to draw, should be in format of torch.Tensor, np.ndarray and list.
    :param use_second: bool, whether to show the figure in format of seconds.
    :param rate: int, the sample rate.
    :return: None
    """
    if isinstance(wav, torch.Tensor):
        if wav.is_cuda:
            wav = wav.cpu()
        wav = wav.numpy()

    if not isinstance(wav, (np.ndarray, list)):
        raise RuntimeError(
            "wav instance can not be drawn, get type {} instead of np.ndarray and list.".format(type(wav)))
    else:
        if isinstance(wav, list):
            wav = np.array(wav)
    if len(wav.shape) == 2:
        wav = np.reshape(wav, (-1,))

    _max = np.abs(wav).max()

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    x = [i for i in range(len(wav))]
    if use_second and rate:
        x = [i / rate for i in x]
    ax.plot(x, wav, color="blue")
    ax.set_title("Wav sound files")
    if sep:
        if isinstance(sep, np.ndarray):
            sep = sep.squeeze()
            for i in range(len(sep)):
                ax.plot([sep[i], sep[i]], [0, _max], color="red")
        elif isinstance(sep, list):
            for i in range(len(sep)):
                ax.plot([sep[i], sep[i]], [0, _max], color="red")

    fig.show()
    plt.close(fig)
