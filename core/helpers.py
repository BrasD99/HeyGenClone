import json
import subprocess
import cv2
from core.temp_manager import TempFileManager
from pydub import AudioSegment
from collections import Counter


def get_duration(filename):
    command = f'ffprobe -i {filename} -show_entries format=duration -v quiet -print_format json'
    output = subprocess.check_output(command, shell=True)
    data = json.loads(output)
    return float(data['format']['duration'])

def format_duration(duration):
    hours, remainder = divmod(int(duration), 3600)
    minutes, seconds = divmod(remainder, 60)
    milliseconds = int((duration - int(duration)) * 1000)
    return f'{hours:02d}:{minutes:02d}:{seconds:02d}.{milliseconds:03d}'

def merge(audio_filename, avi_filename, out_filename):
    audio_duration = get_duration(audio_filename)
    video_duration = get_duration(avi_filename)
    duration = format_duration(video_duration)

    if audio_duration > video_duration:
        temp_manager = TempFileManager()
        temp_wav = temp_manager.create_temp_file(suffix='.wav').name
        command = 'ffmpeg -i {} -ss 00:00:00 -to {} -c copy {}'.format(
            audio_filename, duration, temp_wav
        )
        subprocess.call(command, shell=True)
        audio_filename = temp_wav
    else:
        duration = format_duration(audio_duration)

    command = 'ffmpeg -y -i {} -i {} -ss 00:00:00.000 -to {} -strict -2 -q:v 1 {} -loglevel {}'.format(
        audio_filename, avi_filename, duration, out_filename, 'verbose'
    )
    subprocess.call(command, shell=True)

def to_avi(frames, fps):
    temp_manager = TempFileManager()
    temp_result_avi = temp_manager.create_temp_file(suffix='.avi').name
    first_frame = next(iter(frames))
    frame_h, frame_w = frames[first_frame]['frame'].shape[:-1]

    out = cv2.VideoWriter(
        temp_result_avi, cv2.VideoWriter_fourcc(*'DIVX'), fps, (frame_w, frame_h)
    )

    for frame in frames.values():
        frame = cv2.cvtColor(frame['frame'], cv2.COLOR_BGR2RGB)
        out.write(frame)
    out.release()

    return temp_result_avi

def find_person_id(frame_id, speakers, fps):
    for speaker in speakers:
        if 'id' in speaker:
            if int(speaker['start'] * fps) <= frame_id and int(speaker['end'] * fps) >= frame_id:
                return speaker['id']
    return None

def to_extended_frames(frames, speakers, fps, get_face_on_frame):
    extended_frames = dict()

    for frame_id, frame_dict in frames.items():
        person_id = find_person_id(frame_id, speakers, fps)
        extended_frames[frame_id] = {
            'frame': frame_dict['frame'],
            'has_face': False
        }
        if person_id:
            face_dict = get_face_on_frame(person_id, frame_id)
            if face_dict:
                extended_frames[frame_id]['has_face'] = True
                extended_frames[frame_id]['face'] = face_dict['face']
                extended_frames[frame_id]['bbox'] = face_dict['bbox']
            
    return extended_frames

def get_voice_segments(speakers):
    segments = []
    for speaker in speakers:
        segments.append((speaker['start'], speaker['end']))
    return segments

def find_speaker(groups):
    if groups:
        counter = Counter(groups)
        return counter.most_common(1)[0][0]
    return None

def merge_voices(transcriptions, voice_audio):
    speakers_dict = dict()

    for transcription in transcriptions:
        if 'id' in transcription:
            if not transcription['id'] in speakers_dict:
                speakers_dict[transcription['id']] = AudioSegment.silent(duration=0)
            sub_voice = voice_audio[transcription['start'] * 1000: transcription['end'] * 1000]
            speakers_dict[transcription['id']] += sub_voice
        
    return speakers_dict

def get_timestaps(words):
    if words:
        first_word = words[0]
        last_word = words[-1]
        return first_word['start'], last_word['end']
    else:
        0.0, 0.0

def to_segments(updates, audio_duration):
    segments = []
    prev_end = 0

    for i, update in enumerate(updates):
        start = update['start']
        end = update['end']
        voice = update['voice']

        if start > prev_end:
            segments.append({'start': prev_end, 'end': start, 'empty': True})
        
        segments.append({'start': start, 'end': end, 'empty': False, 'voice': voice })

        if i + 1 == len(updates) and end < audio_duration:
            segments.append({'start': end, 'end': audio_duration, 'empty': True})
        
        prev_end = end

    return segments