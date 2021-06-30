import numpy as np
from scipy.io import wavfile





##################################################################
##########################  Attention!  ##########################
# The following functions are not used in the main files


def add_gaussian_white_noise(data: np.ndarray, var: float = 1e-4):
    """
    Add white Gaussian noise to signal
    If no variance is given, simply add jitter.
    Jitter helps eliminate all-zero values.
    :param data: np.ndarray, the original signal
    :param var: float, the added variance
    :return np.ndarray: the processed data
    """
    np.random.seed(0)
    noise = np.random.normal(0, var, len(data))
    return data + noise


def read_wav(filename: str, norm: bool = False, add_noise: bool = False):
    """
    read wav file.
    Normalizes signal to values between -1 and 1.
    Also add some jitter to remove all-zero segments.
    :param filename: str, path to the target sound file
    :param norm: boolean, whether to norm the wav
    :param add_noise: boolean, whether to add the noise
    :returns (int, np.ndarray): the sample rate and the data
    """
    rate, data = wavfile.read(filename)
    if norm:
        data = np.array(data) / float(max(abs(data)))
    if add_noise:
        data = add_gaussian_white_noise(data)
    return rate, data


def write_wav(filename: str, data: np.ndarray, rate: int = 16000):
    """
    Write the wav data into a file
    :param filename: str, path to the sound file
    :param data: np.ndarray, the sound data
    :param rate: int, the sample rate
    :return: None
    """
    wavfile.write(filename, rate, data)


def enframe(x, win_len, hop_len):
    """
    receives a 1D numpy array and divides it into frames.
    outputs a numpy matrix with the frames on the rows.
    :param x:
    :param win_len
    :param hop_len
    """
    x = np.squeeze(x)
    if x.ndim != 1:
        raise TypeError("enframe input must be a 1-dimensional array.")
    n_frames = 1 + np.int(np.floor((len(x) - win_len) / float(hop_len)))
    x_framed = np.zeros((n_frames, win_len))
    for i in range(n_frames):
        x_framed[i] = x[i * hop_len: i * hop_len + win_len]
    return x_framed


def deframe(x_framed, win_len, hop_len):
    '''
    interpolates 1D data with framed alignments into persample values.
    This function helps as a visual aid and can also be used to change
    frame-rate for features, e.g. energy, zero-crossing, etc.
    :param x_framed
    :param win_len
    :param hop_len
    '''
    n_frames = len(x_framed)
    n_samples = (n_frames - 1) * hop_len + win_len
    x_samples = np.zeros((n_samples, 1))
    for i in range(n_frames):
        x_samples[i * hop_len: i * hop_len + win_len] = x_framed[i]
    return x_samples


def compute_energy(xframes):
    """

    :param xframes:
    :return:
    """
    n_frames = xframes.shape[1]
    return np.diagonal(np.dot(xframes, xframes.T)) / float(n_frames)


def compute_log_nrg(xframes):
    """

    :param xframes:
    :return:
    """
    n_frames = xframes.shape[1]
    raw_nrgs = np.log(compute_energy(xframes + 1e-5)) / float(n_frames)
    return (raw_nrgs - np.mean(raw_nrgs)) / (np.sqrt(np.var(raw_nrgs)))


def power_spectrum(xframes):
    """

    :param xframes: input signal, each row is one frame
    """
    X = np.fft.fft(xframes, axis=1)
    X = np.abs(X[:, :X.shape[1] / 2]) ** 2
    return np.sqrt(X)


def zero_mean(xframes):
    """
    remove mean of framed signal
    return zero-mean frames.
    :param xframes
    """
    m = np.mean(xframes, axis=1)
    xframes = xframes - np.tile(m, (xframes.shape[1], 1)).T
    return xframes


def energy_vad(xframes, percent_thr, nrg_thr=0., context=5):
    """
    Picks frames with high energy as determined by a
    user defined threshold.

    This function also uses a 'context' parameter to
    resolve the fluctuative nature of thresholding.
    context is an integer value determining the number
    of neighboring frames that should be used to decide
    if a frame is voiced.

    The log-energy values are subject to mean and var
    normalization to simplify the picking the right threshold.
    In this framework, the default threshold is 0.0
    """
    xframes = zero_mean(xframes)
    n_frames = xframes.shape[0]

    # Compute per frame energies:
    xnrgs = compute_log_nrg(xframes)
    xvad = np.zeros((n_frames, 1))
    for i in range(n_frames):
        start = max(i - context, 0)
        end = min(i + context, n_frames - 1)
        n_above_thr = np.sum(xnrgs[start:end] > nrg_thr)
        n_total = end - start + 1
        xvad[i] = 1. * ((float(n_above_thr) / n_total) > percent_thr)
    return xvad
