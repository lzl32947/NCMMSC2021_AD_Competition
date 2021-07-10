import random
from typing import List, Optional, Dict

import numpy as np
import torch
import torchaudio
from torch.utils.data.dataset import Dataset
import os
import librosa

from configs.types import ADType, AudioFeatures


class AldsDataset(Dataset):
    def __init__(self, use_features: List[AudioFeatures], use_merge: bool = True, crop_count: int = 1,
                 random_disruption: bool = False, sample_length: int = 15, sr: int = 16000,
                 configs: Optional[Dict] = None):
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
        if random_disruption:
            data, label = self.random_disruption(data, label)
        self.train_list = data
        self.label_list = label
        self.sr = sr
        self.use_features = use_features
        self.configs = configs

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

    def random_disruption(self, data: List, label: List) -> (List, List):
        fix_list = [(i, j) for (i, j) in zip(data, label)]
        random.shuffle(fix_list)
        data = [i[0] for i in fix_list]
        label = [i[1] for i in fix_list]
        return data, label

    def __len__(self) -> int:
        return self.count * self.crop_count

    def resample_wav(self, file_path: str, sample_length: int, sr: int) -> np.float32:

        waveform, sample_rate = librosa.load(file_path, sr=sr)
        start = int(random.random() * (len(waveform) - sample_length * sample_rate))
        cropped = waveform[start: start + sample_length * sample_rate]

        return cropped

    def spec(self, input_wav: np.ndarray) -> np.ndarray:
        n_fft = self.configs['n_fft']
        hop_length = self.configs['hop_length']
        spec = librosa.core.stft(input_wav, n_fft=n_fft, hop_length=hop_length)
        spec = librosa.amplitude_to_db(np.abs(spec), ref=np.max)
        return spec

    def melspec(self, input_wav: np.ndarray, sr: int) -> np.ndarray:
        n_fft = self.configs['n_fft']
        n_mels = self.configs['n_mels']
        hop_length = self.configs['hop_length']
        melspec = librosa.feature.melspectrogram(y=input_wav,
                                                 sr=sr,
                                                 n_fft=n_fft,
                                                 hop_length=hop_length,
                                                 n_mels=n_mels)

        melspec = librosa.power_to_db(melspec, ref=np.max)
        return melspec

    def mfcc(self, input_wav: np.ndarray, sr: int) -> np.ndarray:
        n_fft = self.configs['n_fft']
        n_mfcc = self.configs['n_mfcc']
        hop_length = self.configs['hop_length']
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
        output_list = []

        if AudioFeatures.MFCC in self.use_features:
            mfcc_out = self.mfcc(cropped_wav, self.sr)
            output_list.append(mfcc_out)
        if AudioFeatures.SPECS in self.use_features:
            spec_out = self.spec(cropped_wav)
            output_list.append(spec_out)
        if AudioFeatures.MELSPECS in self.use_features:
            melspec_out = self.melspec(cropped_wav, self.sr)
            output_list.append(melspec_out)
        output_list.append(label)
        return output_list


def audio_collate_fn(batch):
    sizes = len(batch)
    collate_list = [[] for i in range(sizes)]
    for index, item in enumerate(batch):
        collate_list[index].append(item[index])
    return collate_list
