import argparse
import requests
import json
from bs4 import BeautifulSoup
from core.voice_cloner import VoiceCloner
from core.dereverb import MDXNetDereverb
from core.face.lipsync import LipSync
from core.face.detector import FaceDetector
from core.sst import WhisperExtension
from core.helpers import read_video, to_segments, to_extended_frames, to_avi, merge
from core.translator import TextHelper
from core.audio import split_audio_on_silence, speedup_audio, combine_audio
from core.temp_manager import TempFileManager
from pydub import AudioSegment

with open('config.json', 'r') as f:
    config = json.load(f)

LANGUAGES_URL = config['LANGUAGES_URL']

def contains_only_ascii(input_string):
    return all(ord(char) < 128 for char in input_string)

def get_iso_languages():
    response = requests.get(LANGUAGES_URL)
    soup = BeautifulSoup(response.text, 'html.parser')

    p_tags = soup.find_all('p')

    iso_language_dict = {}

    for p_tag in p_tags[1:]:  # Skipping the first <p> which contains the header
        parts = p_tag.get_text().split()
        if len(parts) == 2:
            iso_code, language_name = parts
            if contains_only_ascii(language_name):
                iso_language_dict[language_name] = iso_code

    return iso_language_dict

def translate(video_filename, output_language, output_filename):
    # Models init
    cloner = VoiceCloner(output_language)
    dereverb = MDXNetDereverb(15)
    lip_sync = LipSync()
    face_detector = FaceDetector(config)
    whisper = WhisperExtension()
    text_helper = TextHelper()

    temp_manager = TempFileManager()

    video_params = read_video(video_filename)

    # Splitting audio to speech and noise
    outputs = dereverb.split(video_params['audio'])
    # Splitting voice audio to chunks by silence
    #chunks = split_audio_on_silence(outputs['voice_file'], silence_thresh=-40)
    chunk_audio = AudioSegment.from_file(outputs['voice_file'], format='wav')
    chunks = [{
        'start': 0,
        'end': chunk_audio.duration_seconds * 1000,
        'filename': outputs['voice_file']
    }]

    updates = []

    for chunk in chunks:
        # Getting language and text from speech
        lang, text_arr = whisper.to_text(chunk['filename'])

        for text_dict in text_arr:
            if text_dict['text']:
                # Translating to another lang
                dst_text = text_helper.translate(text_dict['text'], src_lang=lang, dst_lang=output_language[:-1])
                original_audio = AudioSegment.from_file(chunk['filename'], format="wav")
                subaudio = original_audio[text_dict['start'] * 1000:text_dict['end'] * 1000]
                subaudio_wav = temp_manager.create_temp_file(suffix='.wav').name
                subaudio.export(subaudio_wav, format="wav")
                # Cloning voice and reading this text
                cloned_wav = cloner.process(
                    speaker_wav_filename=subaudio_wav,
                    text=dst_text
                )
                # Changing speed of audio to fit duration of original one
                output_wav = speedup_audio(cloned_wav, subaudio_wav)
                # Adding results to updates
                updates.append({
                    # In ms
                    'start': chunk['start'] + text_dict['start'] * 1000,
                    'end': chunk['start'] + text_dict['end'] * 1000,
                    'voice': output_wav
                })
    
    original_audio_duration = AudioSegment.from_file(video_params['audio']) \
        .duration_seconds * 1000
    
    # Getting segments of audio with silence
    segments = to_segments(updates, original_audio_duration)

    # Creating audio without noise
    speech_audio = AudioSegment.silent(duration=0)
    for segment in segments:
        if segment['empty']:
            duration = segment['end'] - segment['start']
            speech_audio += AudioSegment.silent(duration=duration)
        else:
            speech_audio += AudioSegment.from_file(segment['voice'])
    
    speech_audio_wav = temp_manager.create_temp_file(suffix='.wav').name
    speech_audio.export(speech_audio_wav, format='wav')

    frames = to_extended_frames(video_params['frames'], face_detector, config['DET_TRESH'])
    frames = lip_sync.sync(frames, speech_audio_wav, video_params['fps'])
    temp_result_avi = to_avi(frames, video_params['fps'])

    # Combining the resulting audio + original noise
    combined_audio = combine_audio(speech_audio_wav, outputs['noise_file'])

    # Getting all together and saving mp4
    merge(combined_audio, temp_result_avi, output_filename)

if __name__ == '__main__':
    langs = get_iso_languages()
    parser = argparse.ArgumentParser(description='Combine an audio file and a video file into a new video file')
    parser.add_argument('video_filename', help='path to video file')
    parser.add_argument('output_language', choices=list(langs.values()), default='rus', help='choose one option')
    parser.add_argument('-o', '--output_filename', default='output.mp4', help='output file name (default: output.mp4)')
    args = parser.parse_args()

    translate(
        video_filename=args.video_filename,
        output_language=args.output_language,
        output_filename=args.output_filename
    )