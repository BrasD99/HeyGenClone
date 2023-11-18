from pydub import AudioSegment
from pydub.silence import split_on_silence

from audiostretchy.stretch import stretch_audio
from core.temp_manager import TempFileManager

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

    src_chunks = split_on_silence(src, min_silence_len=500, silence_thresh=-40)
    chunks_duration = sum(chunk.duration_seconds for chunk in src_chunks)
    audio = AudioSegment.silent(duration=0)

    if dst_duration >= src_duration:
        silence_interval = (dst_duration - chunks_duration) / len(src_chunks)
    else:
        silence_interval = (src_duration - chunks_duration) / len(src_chunks) * 0.1
    
    for chunk in src_chunks:
        audio += chunk
        audio += AudioSegment.silent(duration=silence_interval * 1000)

    temp_manager = TempFileManager()
    temp_file = temp_manager.create_temp_file(suffix='.wav').name
    audio_file = temp_manager.create_temp_file(suffix='.wav').name
    audio.export(audio_file, format='wav')

    ratio = dst_duration / audio.duration_seconds

    stretch_audio(audio_file, temp_file, ratio=ratio)

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
