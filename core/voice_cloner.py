from TTS.api import TTS
from core.temp_manager import TempFileManager
from core.mapper import map
import torch


class VoiceCloner:
    def __init__(self, lang):
        self.lang_code = map(lang)
        self.tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2", gpu=torch.cuda.is_available())

    def process(self, speaker_wav_filename, text, out_filename=None):
        temp_manager = TempFileManager()
        if not out_filename:
            out_filename = temp_manager.create_temp_file(suffix='.wav').name
        self.tts.tts_to_file(
            text=text, 
            speaker_wav=speaker_wav_filename, 
            language=self.lang_code, 
            file_path=out_filename
        )
        return out_filename
