from enum import Enum


class ADType(Enum):
    AD = "AD"  # 重度患者
    HC = "HC"  # 正常患者
    MCI = "MCI"  # 轻度患者


class VoiceType(Enum):
    PIC = "talk_with_pic"
    FRQ = "frequency_talk"
    FTL = "free_talk"


class OutputType(Enum):
    FMCC = "FMCC"
    SPECS = "Spectrogram"
    MELSPECS = "MelSpectrogram"
