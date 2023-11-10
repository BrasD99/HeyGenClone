from TTS.api import TTS
import torch

device = torch.device('cpu')

tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)

tts.tts_to_file(text="Hello world!", speaker_wav="/Users/brasd99/Music/Music/Media.localized/Music/Unknown Artist/Unknown Album/tmp2zhg2a1w.wav", language="ru", file_path="output.wav")