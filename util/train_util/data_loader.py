import random
from typing import List, Optional, Dict, Union, Tuple, Any

import numpy as np
from PIL import Image
import torchaudio
from torch.utils.data.dataset import Dataset
import os
import librosa
from wav_process.unsupervised_vad import *
from configs.types import ADType, AudioFeatures, DatasetMode


class AldsDataset(Dataset):
    """
    This is the dataset overwrite the torch.utils.data.dataset.Dataset and create the dataset for this program
    """

    def __init__(self, use_features: List[AudioFeatures], use_merge: bool = True, use_vad: bool = False,
                 repeat_times: int = 1, random_disruption: bool = False, configs: Dict = None, k_fold: int = 0,
                 current_fold: Optional[int] = None, run_for: Optional[DatasetMode] = DatasetMode.TRAIN) -> None:
        """
        Init the Dataset with the given parameters
        :param use_features: List[AudioFeatures], the features to use and to be processed in this dataset
        :param use_merge: bool, whether to use the merge audios in dataset
        :param use_vad: bool, whether to use the vad output
        :param repeat_times: int, the re-randomized sample counts
        :param random_disruption: int, the re-sample randomized time, given in seconds, e.g. 5s should be given '5'
        :param configs: Dict, the configs dict
        :param k_fold: int, the number to use k-fold validation, 0 by default and mean do not use k-fold validation
        :param current_fold: int, the current fold, used when k-fold is enable(k_fold > 0)
        :param run_for: DatasetMode, default is DatasetMode.Train and determine the aim of the dataset
        """
        # Set audio backend for torch if used
        torchaudio.set_audio_backend("soundfile")
        # Init the target dict
        self.target_dic = {}
        for t in ADType:
            item = t.value
            self.target_dic[item] = []
        # Init the files and set the count of the files
        self.count = 0
        self.init_files(use_merge)
        # Assert the repeat_times should be larger than zero
        assert repeat_times > 0
        self.repeat_times = repeat_times
        self.use_merge = use_merge
        self.use_vad = use_vad
        self.sample_length = configs['crop_length']
        # Transform the dict to list
        data, label = self.dict2list(k_fold, current_fold, run_for)
        if random_disruption:
            data, label = self.random_disruption(data, label)
        self.train_list = data
        self.label_list = label
        self.sr = configs['sr']
        self.use_features = use_features

        self.configs = configs

    def init_files(self, use_merge: bool) -> None:
        """
        Collect all the files into dict and return the count of the dataset
        :param use_merge: bool, whether to use the merge audio files
        """
        # Merge files should be set to 'dataset/merge' by default and original files should be set to 'dataset/raw'
        if use_merge:
            data_dir = os.path.join("dataset", "merge")
        else:
            data_dir = os.path.join("dataset", "raw")
        count = 0
        # Traverse the ADType to collect all the files and calculate the count of the files
        for t in ADType:
            item = t.value
            target_path = os.path.join(data_dir, item)
            for files in os.listdir(target_path):
                # Save the file names to the target_dic
                self.target_dic[item].append(os.path.join(target_path, files))
                count += 1
        self.count = count

    def dict2list(self, k_fold: int, current_fold: Optional[int], run_for: DatasetMode) -> [int, List,
                                                                                            List]:
        """
        Transform the target_dic to list format ,extract the k_fold data and then randomize them
        :param k_fold: int, determine how many splits should the data be set to
        :param current_fold: int, the current fold
        :param run_for: DatasetMode.
            If DatasetMode.Train is set, remove the data which have the congruence relation with current fold
            If DatasetMode.Test is set, keep only the data which have the congruence relation with current fold
        :return: (int, List, List), mean the count of the selected files, the file names and their labels
        """
        target_file_list = []
        label = []
        current_label = 0
        # Flat the dict to list
        for key in self.target_dic.keys():
            file_list = self.target_dic[key]
            for files in file_list:
                target_file_list.append(files)
                label.append(current_label)
            current_label += 1
        assert len(target_file_list) == len(label)
        assert len(label) == self.count

        # The k_fold is used
        if k_fold != 0:
            assert current_fold is not None
            assert current_fold < k_fold
            assert run_for is not None
            if run_for == DatasetMode.TRAIN:
                # Keep the data without the congruence relation with current fold
                target_file_list = [item for index, item in enumerate(target_file_list) if
                                    index % k_fold != current_fold]
                label = [item for index, item in enumerate(label) if
                         index % k_fold != current_fold]
                assert len(target_file_list) == len(label)
                count = len(label)
            elif run_for == DatasetMode.TEST:
                # Keep the data with the congruence relation with current fold
                target_file_list = [item for index, item in enumerate(target_file_list) if
                                    index % k_fold == current_fold]
                label = [item for index, item in enumerate(label) if
                         index % k_fold == current_fold]
                assert len(target_file_list) == len(label)
                count = len(label)
            else:
                raise RuntimeError()
            self.count = count
        return target_file_list, label

    @staticmethod
    def random_disruption(data: List, label: List) -> (List, List):
        """
        Randomize the data.
        :param data: List, the data
        :param label: List, the label
        :return: (List, List), the data and the label
        """
        # Zip the data and label for shuffling
        fix_list = [(i, j) for (i, j) in zip(data, label)]
        # Randomize
        random.shuffle(fix_list)
        # Unzip the data and label
        data = [i[0] for i in fix_list]
        label = [i[1] for i in fix_list]
        return data, label

    def __len__(self) -> int:
        """
        Return the length of the dataset
        :return: int, the length of the dataset
        """
        # The repeat_times mean the re-randomized sample times
        return self.count * self.repeat_times

    def resample_wav(self, file_path: str, sample_length: int, sr: int, use_vad: bool) -> Union[
        Tuple[Any, np.ndarray], Any]:
        """
        Crop part of the audio and return
        :param use_vad: bool, if True, extra audio will be return
        :param file_path: str, the path to the audio file
        :param sample_length: int, the sample length, and be in format of seconds, e.g. 5s
        :param sr: int, the sample rate
        :return: np.ndarray, the cropped audio
        """
        waveform, sample_rate = librosa.load(file_path, sr=sr)
        start = int(random.random() * (len(waveform) - sample_length * sample_rate))
        cropped = waveform[start: start + sample_length * sample_rate]
        if use_vad:
            vad_out = self.pause(waveform, sample_rate, self.configs['vad'])
            vad_cropped = vad_out[start: start + sample_length * sample_rate]
            return cropped, vad_cropped
        else:
            return cropped

    @staticmethod
    def pre_emphasis(signal: np.ndarray, configs: Dict) -> np.ndarray:
        """
        The pre-emphasis procedure
        :param signal: np.ndarray, the audio
        :param configs: Dict, the config file
        :return: np.ndarray, the pre-emphasised audio
        """
        pre = np.append(signal[0], signal[1:] - configs['coefficient'] * signal[:-1])
        return pre

    @staticmethod
    def spec(input_wav: np.ndarray, configs: Dict, normalized: bool = True) -> np.ndarray:
        """
        Generate the Spectrogram of the given audio
        :param input_wav: np.ndarray, the audio files
        :param configs: Dict, the configs
        :param normalized: bool, whether to normalized the audio with mean equals 0 and std equals 1
        :return: np.ndarray, the Spectrogram data
        """
        n_fft = configs['n_fft']
        hop_length = configs['hop_length']
        # Perform the Short-time Fourier transform (STFT)
        spec = librosa.core.stft(input_wav, n_fft=n_fft, hop_length=hop_length)
        # Convert an amplitude spectrogram to dB-scaled spectrogram
        spec = librosa.amplitude_to_db(np.abs(spec), ref=np.max)
        # Normalize the data
        if normalized:
            spec = (spec - spec.mean()) / spec.std()
        # Resize the data to the target shape with the help of PIL.Image
        if configs['resize']:
            resize_height = spec.shape[0] if configs['resize_height'] < 0 else configs['resize_height']
            resize_width = spec.shape[1] if configs['resize_width'] < 0 else configs['resize_width']
            resize_shape = (resize_width, resize_height)
            image = Image.fromarray(spec)

            image = image.resize(resize_shape, Image.ANTIALIAS)
            spec = np.array(image)
        # Expand dimension to 3 to process it as the image
        spec = np.expand_dims(spec, axis=0)
        return spec

    @staticmethod
    def melspec(input_wav: np.ndarray, sr: int, configs: Dict, normalized: bool = True) -> np.ndarray:
        """
        Generate the Mel-Spectrogram of the given audio
        :param input_wav: np.ndarray, the audio files
        :param sr: int, sample rate
        :param configs: Dict, the configs
        :param normalized: bool, whether to normalized the audio with mean equals 0 and std equals 1
        :return: np.ndarray, the Mel-Spectrogram data
        """
        n_fft = configs['n_fft']
        n_mels = configs['n_mels']
        hop_length = configs['hop_length']
        # Compute a mel-scaled spectrogram
        melspec = librosa.feature.melspectrogram(y=input_wav,
                                                 sr=sr,
                                                 n_fft=n_fft,
                                                 hop_length=hop_length,
                                                 n_mels=n_mels)
        # Convert a power spectrogram (amplitude squared) to decibel (dB) units
        melspec = librosa.power_to_db(melspec, ref=np.max)
        # Normalize the data
        if normalized:
            melspec = (melspec - melspec.mean()) / melspec.std()
        # Resize the data to the target shape with the help of PIL.Image
        if configs['resize']:
            resize_height = melspec.shape[0] if configs['resize_height'] < 0 else configs['resize_height']
            resize_width = melspec.shape[1] if configs['resize_width'] < 0 else configs['resize_width']
            resize_shape = (resize_width, resize_height)
            image = Image.fromarray(melspec)

            image = image.resize(resize_shape, Image.ANTIALIAS)
            melspec = np.array(image)
        # Expand dimension to 3 to process it as the image
        melspec = np.expand_dims(melspec, axis=0)
        return melspec

    @staticmethod
    def mfcc(input_wav: np.ndarray, sr: int, configs: Dict, normalized: bool = True) -> np.ndarray:
        """
        Generate the MFCC features of the given audio
        :param input_wav: np.ndarray, the audio files
        :param sr: int, sample rate
        :param configs: Dict, the configs
        :param normalized: bool, whether to normalized the audio with mean equals 0 and std equals 1
        :return: np.ndarray, the MFCC features
        """
        n_fft = configs['n_fft']
        n_mfcc = configs['n_mfcc']
        n_mels = configs['n_mels']
        hop_length = configs['hop_length']
        # Calculate the Mel-frequency cepstral coefficients (MFCCs)
        mfcc = librosa.feature.mfcc(input_wav,
                                    sr=sr,
                                    n_fft=n_fft,
                                    n_mfcc=n_mfcc,
                                    n_mels=n_mels,
                                    hop_length=hop_length)
        # Normalize the data
        if normalized:
            mfcc = (mfcc - mfcc.mean()) / mfcc.std()
        # Resize the data to the target shape with the help of PIL.Image
        if configs['resize']:
            resize_height = mfcc.shape[0] if configs['resize_height'] < 0 else configs['resize_height']
            resize_width = mfcc.shape[1] if configs['resize_width'] < 0 else configs['resize_width']
            resize_shape = (resize_width, resize_height)
            image = Image.fromarray(mfcc)

            image = image.resize(resize_shape, Image.ANTIALIAS)
            mfcc = np.array(image)
        # Expand dimension to 3 to process it as the image
        mfcc = np.expand_dims(mfcc, axis=0)
        return mfcc

    @staticmethod
    def pause(input_wav: np.ndarray, sr: int, configs: Dict, normalized: bool = True) -> np.ndarray:
        """
        Generate the pause features of the given audio
        :param sr: int, sample rate
        :param input_wav: np.ndarray, the audio files
        :param configs: Dict, the configs
        :param normalized: bool, whether to normalized the audio with mean equals 0 and std equals 1
        :return: np.ndarray, the pause features
        """
        window_length = int(sr * 0.025)
        hop_length = int(sr * 0.010)
        # En-frame the data
        en_frame = np.squeeze(input_wav)
        n_frames = 1 + int(np.floor((len(en_frame) - window_length) / float(hop_length)))
        data_frame = np.zeros((n_frames, window_length))
        for i in range(n_frames):
            data_frame[i] = en_frame[i * hop_length: i * hop_length + window_length]

        percent_high_nrg = configs['percent_high_nrg']

        # Calculate zero frames
        zero_mean_frames = data_frame - np.tile(np.mean(data_frame, axis=1), (data_frame.shape[1], 1)).T

        # Calculate frame energy
        threshold = 1e-5
        frame_energy = np.diagonal(np.dot(zero_mean_frames + threshold, (zero_mean_frames + threshold).T)) / float(
            zero_mean_frames.shape[1])
        # Calculate frame log energy
        raw_log_energy = np.log(frame_energy) / float(zero_mean_frames.shape[1])
        log_energy = (raw_log_energy - np.mean(raw_log_energy)) / (np.sqrt(np.var(raw_log_energy)))

        # Calculate the VAD
        context = 5
        energy_threshold = 0
        vad_output = np.zeros((n_frames, 1))
        for i in range(n_frames):
            start = max(i - context, 0)
            end = min(i + context, n_frames - 1)
            n_above_thr = np.sum(log_energy[start:end] > energy_threshold)
            n_total = end - start + 1
            vad_output[i] = 1. * ((float(n_above_thr) / n_total) > percent_high_nrg)

        # De-frame the data
        n_samples = (n_frames - 1) * hop_length + window_length
        sample_output = np.zeros((n_samples, 1))
        for i in range(n_frames):
            sample_output[i * hop_length: i * hop_length + window_length] = vad_output[i]

        # Generate the output
        sample_output = sample_output.squeeze()
        sample_output = np.pad(sample_output, (0, len(input_wav) - len(sample_output)), 'constant', constant_values=0.)
        processed_wav = sample_output * input_wav
        return processed_wav

    def __getitem__(self, item: int) -> Dict:
        """
        Get one item from the dataset
        :param item: int, the order of the given data
        :return: Dict, the data
        """
        # Get the file and the label
        file = self.train_list[item % self.count]
        label = self.label_list[item % self.count]
        output_dict = {}
        # Read and crop the audio
        if self.use_vad:

            cropped_wav, vad_cropped = self.resample_wav(file, self.sample_length, self.sr, self.use_vad)
            # Pre-emphasis the audio
            output_wav = self.pre_emphasis(cropped_wav, self.configs['pre_emphasis'])
            output_vad = self.pre_emphasis(vad_cropped, self.configs['pre_emphasis'])
        else:
            cropped_wav: np.ndarray = self.resample_wav(file, self.sample_length, self.sr, self.use_vad)
            # Pre-emphasis the audio
            output_wav = self.pre_emphasis(cropped_wav, self.configs['pre_emphasis'])
            output_vad = None

        for item in self.use_features:
            # Add the MFCC feature to output if used
            if AudioFeatures.MFCC == item:
                mfcc_out = self.mfcc(output_wav, self.sr, self.configs['mfcc'],
                                     normalized=self.configs['normalized'])
                output_dict[AudioFeatures.MFCC] = mfcc_out
            # Add the Spectrogram feature to output if used
            if AudioFeatures.SPECS == item:
                spec_out = self.spec(output_wav, self.configs['specs'], normalized=self.configs['normalized'])
                output_dict[AudioFeatures.SPECS] = spec_out
            # Add the Mel-Spectrogram feature to output if used
            if AudioFeatures.MELSPECS == item:
                melspec_out = self.melspec(output_wav, self.sr, self.configs['melspecs'],
                                           normalized=self.configs['normalized'])
                output_dict[AudioFeatures.MELSPECS] = melspec_out
        if self.use_vad:
            assert output_vad is not None
            for item in self.use_features:

                # Add the MFCC feature to output if used
                if AudioFeatures.MFCC == item:
                    mfcc_out = self.mfcc(output_vad, self.sr, self.configs['mfcc'],
                                         normalized=self.configs['normalized'])
                    output_dict[AudioFeatures.MFCC_VAD] = mfcc_out
                # Add the Spectrogram feature to output if used
                if AudioFeatures.SPECS == item:
                    spec_out = self.spec(output_vad, self.configs['specs'], normalized=self.configs['normalized'])
                    output_dict[AudioFeatures.SPECS_VAD] = spec_out
                # Add the Mel-Spectrogram feature to output if used
                if AudioFeatures.MELSPECS == item:
                    melspec_out = self.melspec(output_vad, self.sr, self.configs['melspecs'],
                                               normalized=self.configs['normalized'])
                    output_dict[AudioFeatures.MELSPECS_VAD] = melspec_out

        # Add the label to output
        output_dict[AudioFeatures.LABEL] = label
        # Add raw audio to output
        output_dict[AudioFeatures.RAW] = output_wav
        # Add vad audio to output
        if self.use_vad:
            output_dict[AudioFeatures.VAD] = output_vad
        return output_dict


def audio_collate_fn(batch):
    # The custom the collate_fn function
    # NOTICE: this function is not used in program and not been test
    sizes = len(batch)
    collate_list = [[] for i in range(sizes)]
    for index, item in enumerate(batch):
        collate_list[index].append(item[index])
    return collate_list
