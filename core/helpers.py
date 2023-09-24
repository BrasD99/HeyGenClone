import json
import subprocess
import os
import tempfile
import cv2
from moviepy.video.io.VideoFileClip import VideoFileClip
from core.temp_manager import TempFileManager

def read_config(file='config.json'):
    with open(file, 'r') as f:
        return json.load(f)

def read_video(filename):
    clip = VideoFileClip(filename)
    frames = []
    for frame in clip.iter_frames():
        frames.append(frame)
    
    temp_manager = TempFileManager()
    temp_file = temp_manager.create_temp_file(suffix='.wav').name
    clip.audio.write_audiofile(temp_file, codec='pcm_s16le')

    return {
        'fps': clip.fps,
        'frames': frames,
        'audio': temp_file
    }

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

def create_temp_file(ext):
    temp, filename = tempfile.mkstemp()
    os.close(temp)
    return f'{filename}.{ext}'

def merge(audio_filename, avi_filename, out_filename):
    audio_duration = get_duration(audio_filename)
    video_duration = get_duration(avi_filename)
    duration = format_duration(video_duration)

    if audio_duration > video_duration:
        temp_wav = create_temp_file('wav')
        command = f'ffmpeg -i {audio_filename} -ss 00:00:00 -to {duration} -c copy {temp_wav}'
        subprocess.call(command, shell=True)
        audio_filename = temp_wav
    else:
        duration = format_duration(audio_duration)

    command = 'ffmpeg -y -i {} -i {} -ss 00:00:00.000 -to {} -strict -2 -q:v 1 {} -loglevel {}'.format(
        audio_filename, avi_filename, duration, out_filename, 'verbose'
    )
    subprocess.call(command, shell=True)

def to_avi(frames, fps):
    temp_result_avi = create_temp_file('avi')
    frame_h, frame_w = frames[0]['frame'].shape[:-1]

    out = cv2.VideoWriter(
        temp_result_avi, cv2.VideoWriter_fourcc(*'DIVX'), fps, (frame_w, frame_h)
    )

    for frame in frames.values():
        frame = cv2.cvtColor(frame['frame'], cv2.COLOR_BGR2RGB)
        out.write(frame)
    out.release()

    return temp_result_avi

def to_extended_frames(frames, face_detector, det_tresh):
    extended_frames = dict()

    for frame_id, frame in enumerate(frames):
        extended_frames[frame_id] = {
            'frame': frame,
            'has_face': False
        }
        faces = face_detector.detect(frame, det_tresh)
        if faces:
            face, coords = faces[0]
            extended_frames[frame_id]['face'] = face
            extended_frames[frame_id]['has_face'] = True
            extended_frames[frame_id]['c'] = coords
    return extended_frames

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