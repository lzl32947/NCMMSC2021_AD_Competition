import random
from typing import List, Optional, Dict

import numpy as np
import torch
import torchaudio
from torch.utils.data.dataset import Dataset
import os
import librosa

from configs.types import ADType, AudioFeatures, DatasetMode


class AldsDataset(Dataset):
    def __init__(self, use_features: List[AudioFeatures], use_merge: bool = True, crop_count: int = 1,
                 random_disruption: bool = False, configs: Dict = None, k_fold: int = 0,
                 current_fold: Optional[int] = None, run_for: Optional[DatasetMode] = DatasetMode.TRAIN):
        torchaudio.set_audio_backend("soundfile")
        self.target_dic = {}
        for t in ADType:
            item = t.value
            self.target_dic[item] = []
        self.count = self.init_files(use_merge)
        assert crop_count > 0
        self.crop_count = crop_count
        self.use_merge = use_merge
        self.sample_length = configs['crop_length']
        self.count, data, label = self.dict2list(self.count, k_fold, current_fold, run_for)
        if random_disruption:
            data, label = self.random_disruption(data, label)
        self.train_list = data
        self.label_list = label
        self.sr = configs['sr']
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

    def dict2list(self, count: int, k_fold: int, current_fold: Optional[int], run_for: DatasetMode):
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

        if k_fold != 0:
            assert current_fold is not None
            assert current_fold < k_fold
            assert run_for is not None
            if run_for == DatasetMode.TRAIN:
                target_file_list = [item for index, item in enumerate(target_file_list) if
                                    index % k_fold != current_fold]
                label = [item for index, item in enumerate(label) if
                         index % k_fold != current_fold]
                assert len(target_file_list) == len(label)
                count = len(label)
            elif run_for == DatasetMode.TEST:
                target_file_list = [item for index, item in enumerate(target_file_list) if
                                    index % k_fold == current_fold]
                label = [item for index, item in enumerate(label) if
                         index % k_fold == current_fold]
                assert len(target_file_list) == len(label)
                count = len(label)
        return count, target_file_list, label

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

    def pre_emphasis(self, signal: np.ndarray, configs: Dict) -> np.ndarray:
        pre = np.append(signal[0], signal[1:] - configs['coefficient'] * signal[:-1])
        return pre

    def spec(self, input_wav: np.ndarray, configs: Dict, normalized: bool = True) -> np.ndarray:
        n_fft = configs['n_fft']
        hop_length = configs['hop_length']
        spec = librosa.core.stft(input_wav, n_fft=n_fft, hop_length=hop_length)
        spec = librosa.amplitude_to_db(np.abs(spec), ref=np.max)
        if normalized:
            spec = (spec - spec.mean()) / spec.std()
        return spec

    def melspec(self, input_wav: np.ndarray, configs: Dict, normalized: bool = True) -> np.ndarray:
        n_fft = configs['n_fft']
        n_mels = configs['n_mels']
        hop_length = configs['hop_length']
        melspec = librosa.feature.melspectrogram(y=input_wav,
                                                 sr=self.configs['sr'],
                                                 n_fft=n_fft,
                                                 hop_length=hop_length,
                                                 n_mels=n_mels)

        melspec = librosa.power_to_db(melspec, ref=np.max)
        if normalized:
            melspec = (melspec - melspec.mean()) / melspec.std()
        return melspec

    def mfcc(self, input_wav: np.ndarray, configs: Dict, normalized: bool = True) -> np.ndarray:
        n_fft = configs['n_fft']
        n_mfcc = configs['n_mfcc']
        hop_length = configs['hop_length']
        mfcc = librosa.feature.mfcc(input_wav,
                                    sr=self.configs['sr'],
                                    n_fft=n_fft,
                                    n_mfcc=n_mfcc,
                                    hop_length=hop_length)
        if normalized:
            mfcc = (mfcc - mfcc.mean()) / mfcc.std()
        return mfcc

    def __getitem__(self, item):
        file = self.train_list[item % self.count]
        label = self.label_list[item % self.count]
        cropped_wav: np.ndarray = self.resample_wav(file, self.sample_length, self.sr)
        output_wav = self.pre_emphasis(cropped_wav, self.configs['pre_emphasis'])
        output_list = []

        if AudioFeatures.MFCC in self.use_features:
            mfcc_out = self.mfcc(output_wav, self.configs['mfcc'], normalized=self.configs['normalized'])
            output_list.append(mfcc_out)
        if AudioFeatures.SPECS in self.use_features:
            spec_out = self.spec(output_wav, self.configs['specs'], normalized=self.configs['normalized'])
            output_list.append(spec_out)
        if AudioFeatures.MELSPECS in self.use_features:
            melspec_out = self.melspec(output_wav, self.configs['melspecs'], normalized=self.configs['normalized'])
            output_list.append(melspec_out)
        output_list.append(label)
        return output_list


def audio_collate_fn(batch):
    sizes = len(batch)
    collate_list = [[] for i in range(sizes)]
    for index, item in enumerate(batch):
        collate_list[index].append(item[index])
    return collate_list
