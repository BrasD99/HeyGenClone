from core.whisperx.asr import load_model, load_audio
from core.whisperx.alignment import load_align_model, align
import torch

class WhisperExtension:
    def __init__(self):
        self.whisper = load_model('large-v2', device='cpu', compute_type='int8')
        self.batch_size = 16
        self.device = torch.device('cpu')
    
    def to_text(self, audio_file):
        audio = load_audio(audio_file)
        result = self.whisper.transcribe(audio, batch_size=self.batch_size)
        language = result['language']
        model_a, metadata = load_align_model(language_code=result["language"], device=self.device)
        result = align(result["segments"], model_a, metadata, audio, self.device, return_char_alignments=False)

        return language, result['segments']