import random

import numpy as np
import torch
import torchaudio
from torch.utils.data.dataset import Dataset
import os
import librosa
from torchaudio.transforms import MFCC

from configs.types import ADType, OutputType


class AldsDataset(Dataset):
    def __init__(self, output_type: [OutputType, ...], use_merge: bool = True, crop_count: int = 1,
                 sample_length: int = 30, sr: int = 22050):
        torchaudio.set_audio_backend("soundfile")
        self.target_dic = {}
        for t in ADType:
            item = t.value
            self.target_dic[item] = []
        self.count = self.init_files(use_merge)
        assert crop_count > 0
        self.crop_count = crop_count
        self.use_merge = use_merge
        self.sample_length = sample_length
        data, label = self.dict2list(self.count)
        self.train_list = data
        self.label_list = label
        self.sr = sr
        self.output_type = output_type

    def init_files(self, use_merge: bool):
        if use_merge:
            data_dir = os.path.join("dataset", "merge")
        else:
            data_dir = os.path.join("dataset", "raw")
        count = 0
        for t in ADType:
            item = t.value
            target_path = os.path.join(data_dir, item)
            for files in os.listdir(target_path):
                self.target_dic[item].append(os.path.join(target_path, files))
                count += 1
        return count

    def dict2list(self, count: int):
        target_file_list = []
        label = []
        current_label = 0
        for key in self.target_dic.keys():
            file_list = self.target_dic[key]
            for files in file_list:
                target_file_list.append(files)
                label.append(current_label)
            current_label += 1
        assert len(target_file_list) == len(label)
        assert len(label) == count
        return target_file_list, label

    def __len__(self):
        return self.count * self.crop_count

    def resample_wav(self, file_path: str, sample_length: int, sr: int) -> np.float32:

        waveform, sample_rate = librosa.load(file_path, sr=sr)
        start = int(random.random() * (len(waveform) - sample_length * sample_rate))
        cropped = waveform[start: start + sample_length * sample_rate]

        return cropped

    def spec(self, input_wav: np.ndarray):
        n_fft = 1024
        hop_length = 512
        spec = librosa.core.stft(input_wav, n_fft=n_fft, hop_length=hop_length)
        spec = librosa.amplitude_to_db(np.abs(spec), ref=np.max)
        return spec

    def melspec(self, input_wav: np.ndarray, sr: int):
        n_fft = 1024
        hop_length = 512
        n_mels = 128
        melspec = librosa.feature.melspectrogram(y=input_wav,
                                                 sr=sr,
                                                 n_fft=n_fft,
                                                 hop_length=hop_length,
                                                 n_mels=n_mels)

        melspec = librosa.power_to_db(melspec, ref=np.max)
        return melspec

    def mfcc(self, input_wav: np.ndarray, sr: int):
        n_fft = 1024
        n_mfcc = 20
        hop_length = 512
        mfcc = librosa.feature.mfcc(input_wav,
                                    sr=sr,
                                    n_fft=n_fft,
                                    n_mfcc=n_mfcc,
                                    hop_length=hop_length)
        return mfcc

    def __getitem__(self, item):
        file = self.train_list[item % self.count]
        label = self.label_list[item % self.count]
        cropped_wav: np.ndarray = self.resample_wav(file, self.sample_length, self.sr)

        mfcc_out = self.mfcc(cropped_wav, self.sr)
        spec_out = self.spec(cropped_wav)
        melspec_out = self.melspec(cropped_wav, self.sr)

        return mfcc_out, spec_out, melspec_out, label
