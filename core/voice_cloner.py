from TTS.api import TTS
from core.temp_manager import TempFileManager
from core.mapper import map_to_tts


class VoiceCloner:
    def __init__(self, lang):
        lang_code = map_to_tts(lang)
        self.api = TTS(f'tts_models/{lang_code}/fairseq/vits')

    def process(self, speaker_wav_filename, text, out_filename=None):
        temp_manager = TempFileManager()
        if not out_filename:
            out_filename = temp_manager.create_temp_file(suffix='.wav').name
        self.api.tts_with_vc_to_file(
            text,
            speaker_wav=speaker_wav_filename,
            file_path=out_filename
        )
        return out_filename
