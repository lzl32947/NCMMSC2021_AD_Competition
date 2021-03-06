#-*- coding: UTF-8 -*-
from enum import Enum


class ADType(Enum):
    AD = "AD"  # 重度患者
    HC = "HC"  # 正常患者
    MCI = "MCI"  # 轻度患者


class VoiceType(Enum):
    PIC = "talk_with_pic"
    FRQ = "frequency_talk"
    FTL = "free_talk"


class AudioFeatures(Enum):
    # Features
    MFCC = "MFCC"
    SPECS = "Spectrogram"
    MELSPECS = "MelSpectrogram"
    # Features from VAD output
    MFCC_VAD = "MFCC_VAD"
    SPECS_VAD = "Spectrogram_VAD"
    MELSPECS_VAD = "MelSpectrogram_VAD"
    # Label
    LABEL = "LABEL"
    # Audio
    RAW = "RAW"
    VAD = "VAD"
    # Name
    NAME = "NAME"


class DatasetMode(Enum):
    TRAIN = "train"
    TEST = "test"
    VALID = "valid"
