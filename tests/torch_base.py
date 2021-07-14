import torchaudio

from configs.types import ADType
from util.tools.draw_util import draw_sound
from util.tools.files_util import *
import os


def audio_file_generator(base_path: str):
    for t in ADType:
        item = t.value
        target_dir = os.path.join(base_path, item)
        for wav_file in os.listdir(target_dir):
            wav_file_path = os.path.join(target_dir, wav_file)
            waveform, rate = torchaudio.load(wav_file_path)
            yield waveform, rate


if __name__ == '__main__':
    set_working_dir("./..")

    torchaudio.set_audio_backend("soundfile")

    for wav, r in audio_file_generator("dataset/merge"):
        draw_sound(wav, rate=r)
