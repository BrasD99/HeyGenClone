from core.dereverb import MDXNetDereverb
from core.face.detector import FaceDetector
from core.face.lipsync import LipSync
from core.helpers import to_avi, merge
from core.temp_manager import TempFileManager
from core.audio import combine_audio
from moviepy.video.io.VideoFileClip import VideoFileClip
from pydub import AudioSegment
import argparse
from tqdm import tqdm
import json


# Only for video with one person
def update_voice(voice_filename, video_filename, output_filename):
    with open('config.json', 'r') as f:
        config = json.load(f)

    dereverb = MDXNetDereverb(15)
    temp_manager = TempFileManager()
    face_detector = FaceDetector()
    lip_sync = LipSync()

    orig_clip = VideoFileClip(video_filename, verbose=False)
    original_audio_file = temp_manager.create_temp_file(suffix='.wav').name
    orig_clip.audio.write_audiofile(
        original_audio_file, codec='pcm_s16le', verbose=False, logger=None)

    dereverb_out = dereverb.split(original_audio_file)
    noise_audio = AudioSegment.from_file(
        dereverb_out['noise_file'], format='wav')

    frames = dict()
    for frame_id, frame in tqdm(enumerate(orig_clip.iter_frames()), desc='Processing frames'):
        frames[frame_id] = {
            'frame': frame,
            'has_face': False
        }
        detections = face_detector.detect(frame, config['DET_TRESH'])
        if detections:
            face, bbox = detections[0]
            frames[frame_id]['has_face'] = True
            frames[frame_id]['face'] = face
            frames[frame_id]['bbox'] = bbox

    frames = lip_sync.sync(frames, voice_filename, orig_clip.fps)
    temp_result_avi = to_avi(frames, orig_clip.fps)

    noise_audio_wav = temp_manager.create_temp_file(suffix='.wav').name
    noise_audio.export(noise_audio_wav, format='wav')

    combined_audio = combine_audio(voice_filename, noise_audio_wav)

    merge(combined_audio, temp_result_avi, output_filename)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Combine an audio file and a video file into a new video file')
    parser.add_argument('voice_filename', help='path to voice file')
    parser.add_argument('video_filename', help='path to video file')
    parser.add_argument('-o', '--output_filename', default='output.mp4',
                        help='output file name (default: output.mp4)')
    args = parser.parse_args()

    update_voice(
        video_filename=args.video_filename,
        voice_filename=args.voice_filename,
        output_filename=args.output_filename
    )
