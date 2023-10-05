import itertools
import os

from pydub import AudioSegment
from pydub.silence import detect_nonsilent

from audiostretchy.stretch import stretch_audio
from core.temp_manager import TempFileManager

def split_on_silence(audio_segment, min_silence_len=1000, silence_thresh=-16, keep_silence=100,
                     seek_step=1):
    def pairwise(iterable):
        's -> (s0,s1), (s1,s2), (s2, s3), ...'
        a, b = itertools.tee(iterable)
        next(b, None)
        return zip(a, b)
    
    if isinstance(keep_silence, bool):
        keep_silence = len(audio_segment) if keep_silence else 0
    
    output_ranges = [
        [ start - keep_silence, end + keep_silence ]
        for (start,end)
            in detect_nonsilent(audio_segment, min_silence_len, silence_thresh, seek_step)
    ]

    for range_i, range_ii in pairwise(output_ranges):
        last_end = range_i[1]
        next_start = range_ii[0]
        if next_start < last_end:
            range_i[1] = (last_end+next_start)//2
            range_ii[0] = range_i[1]

    return [
        {
            'audio': audio_segment[max(start, 0) : min(end, len(audio_segment))],
            'start': max(start,0),
            'end': min(end,len(audio_segment))
        }
        for start, end in output_ranges
    ]

def remove_silence(audio, silence_thresh=-40):
    silence_thresh = float(silence_thresh)  # Convert silence_thresh to float
    non_silent_audio = split_on_silence(audio, min_silence_len=1000, silence_thresh=silence_thresh)
    return non_silent_audio

def join_audio_segments(segments, segment_duration, min_segment_duration):
    joined_segments = []
    current_segment = None
    for segment in segments:
        if current_segment is None:
            current_segment = segment
        elif len(current_segment) < segment_duration * 1000:
            current_segment += segment
        else:
            # Check if the current segment is longer than the minimum duration
            if len(current_segment) >= min_segment_duration * 1000:
                joined_segments.append(current_segment)
            current_segment = segment

    if current_segment is not None:
        # Check if the current segment is longer than the minimum duration
        if len(current_segment) >= min_segment_duration * 1000:
            joined_segments.append(current_segment)

    return joined_segments

def split_audio_on_silence(
        audio_file_path,
        silence_thresh,
        output_format='wav'):
    
    audio = AudioSegment.from_file(audio_file_path)
    non_silent_audio = remove_silence(audio, silence_thresh=silence_thresh)
    
    temp_files = []
    temp_manager = TempFileManager()
    for k, segment in enumerate(non_silent_audio):
        segment_file_name = f'segment_{k + 1}.{output_format}'
        temp_file = temp_manager.create_temp_file(suffix=f'.{output_format}')
        segment['audio'].export(temp_file.name, format=output_format)  # Write segment to the temp file
        temp_file.close()  # Close the temp file
        temp_file_name = os.path.join(os.path.dirname(temp_file.name), segment_file_name)

        # Check if the destination file already exists
        counter = 1
        while os.path.exists(temp_file_name):
            segment_file_name = f'segment_{k + 1}_{counter}.{output_format}'
            temp_file_name = os.path.join(os.path.dirname(temp_file.name), segment_file_name)
            counter += 1

        os.rename(temp_file.name, temp_file_name)  # Rename the temp file
        temp_files.append({
            'start': segment['start'],
            'end': segment['end'],
            'filename': temp_file_name
        })

    return temp_files
    
def speed_change(sound, speed=1.0):
    # Manually override the frame_rate. This tells the computer how many
    # samples to play per second
    sound_with_altered_frame_rate = sound._spawn(sound.raw_data, overrides={
        'frame_rate': int(sound.frame_rate * speed)
    })

    # convert the sound with altered frame rate to a standard frame rate
    # so that regular playback programs will work right. They often only
    # know how to play audio at standard frame rate (like 44.1k)
    return sound_with_altered_frame_rate.set_frame_rate(sound.frame_rate)
    
def speedup_audio(src_audio_filename, dst_audio_filename):
    src = AudioSegment.from_file(src_audio_filename)
    dst = AudioSegment.from_file(dst_audio_filename)

    src_duration = src.duration_seconds
    dst_duration = dst.duration_seconds

    # Fitting to dst
    ratio = dst_duration / src_duration

    temp_manager = TempFileManager()
    temp_file = temp_manager.create_temp_file(suffix='.wav').name
    stretch_audio(src_audio_filename, temp_file, ratio=ratio)

    stretched_audio = AudioSegment.from_file(temp_file)
    cropped_audio = stretched_audio[:dst_duration * 1000]

    cropped_audio.export(temp_file, format='wav')

    return temp_file

def combine_audio(audio_file_1, audio_file_2):
    a1 = AudioSegment.from_wav(audio_file_1)
    a2 = AudioSegment.from_wav(audio_file_2)

    tmpsound = a1.overlay(a2)
    temp_manager = TempFileManager()
    temp_file = temp_manager.create_temp_file(suffix='.wav').name

    tmpsound.export(temp_file, format='wav') 
    return temp_file