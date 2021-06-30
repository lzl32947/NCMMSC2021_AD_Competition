import matplotlib.pyplot as plt
import numpy as np


def draw_sound(wav, sep=None):
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
    ax.plot(wav, color="blue")
    ax.set_title("Wav sound files")
    if sep is not None:
        if isinstance(sep, np.ndarray):
            sep = sep.squeeze()
            for i in range(len(sep)):
                ax.plot([sep[i], sep[i]], [0, _max], color="red")
        elif isinstance(sep, list):
            for i in range(len(sep)):
                ax.plot([sep[i], sep[i]], [0, _max], color="red")

    fig.show()
    plt.close(fig)
