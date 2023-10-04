from scenedetect.video_manager import VideoManager
from scenedetect.scene_manager import SceneManager
from scenedetect.stats_manager import StatsManager
from scenedetect.detectors import ContentDetector

from core.face.detector import FaceDetector
from core.temp_manager import TempFileManager
from core.talkNet.talkNet import talkNet
from moviepy.video.io.VideoFileClip import VideoFileClip

from scipy.interpolate import interp1d
from scipy.io import wavfile
from scipy import signal

from tqdm import tqdm

import numpy as np
import cv2
import subprocess
import torch
import python_speech_features
import math
import os

def bb_intersection_over_union(boxA, boxB, evalCol=False):
    # IOU Function to calculate overlap between two bounding boxes
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[0] + boxA[2], boxB[0] + boxB[2])
    yB = min(boxA[1] + boxA[3], boxB[1] + boxB[3])

    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = boxA[2] * boxA[3]
    boxBArea = boxB[2] * boxB[3]

    if evalCol:
        iou = interArea / float(boxAArea)
    else:
        iou = interArea / float(boxAArea + boxBArea - interArea)

    return iou

def track_shot(scene_faces, num_failed_dets=10, min_track=10, min_face_size=1):
    iouThres  = 0.5
    tracks = []

    while True:
        track = []
        for frameFaces in scene_faces:
            for face in frameFaces:
                if track == []:
                    track.append(face)
                    frameFaces.remove(face)
                elif face['frame'] - track[-1]['frame'] <= num_failed_dets:
                    iou = bb_intersection_over_union(face['bbox'], track[-1]['bbox'])
                    if iou > iouThres:
                        track.append(face)
                        frameFaces.remove(face)
                        continue
                else:
                    break
        if track == []:
            break
        elif len(track) > min_track:
            frameNum = np.array([f['frame'] for f in track])
            bboxes = np.array([np.array(f['bbox']) for f in track])
            frameI = np.arange(frameNum[0], frameNum[-1] + 1)
            bboxesI = []
            for ij in range(0, 4):
                interpfn = interp1d(frameNum, bboxes[:,ij])
                bboxesI.append(interpfn(frameI))
            bboxesI  = np.stack(bboxesI, axis=1)
            if max(np.mean(bboxesI[:,2]-bboxesI[:,0]), np.mean(bboxesI[:,3]-bboxesI[:,1])) > min_face_size:
                tracks.append({'frame': frameI,'bbox': bboxesI})

    return tracks

def find_faces(video_file_path, config):
    face_detector = FaceDetector(config)
    clip = VideoFileClip(video_file_path)
    results = []
    frames = []
    for frame_id, frame in enumerate(clip.iter_frames()):
        results.append([])
        faces = face_detector.detect(frame, face_det_tresh=config['DET_TRESH'])
        for face in faces:
            _, bbox = face
            results[-1].append({'frame': frame_id, 'bbox': bbox})
        frames.append(frame)
    
    temp_manager = TempFileManager()
    audio_temp_file = temp_manager.create_temp_file(suffix='.wav').name
    clip.audio.write_audiofile(audio_temp_file, codec='pcm_s16le')
    return results, frames, audio_temp_file

def detect_scene(video_file_path):
    videoManager = VideoManager([video_file_path])
    statsManager = StatsManager()
    sceneManager = SceneManager(statsManager)
    sceneManager.add_detector(ContentDetector())
    baseTimecode = videoManager.get_base_timecode()
    videoManager.set_downscale_factor()
    videoManager.start()
    sceneManager.detect_scenes(frame_source = videoManager)
    return sceneManager.get_scene_list(baseTimecode)

def crop_video(track, audio_file_path, image, loader_threads=10, crop_scale=0.40):
    temp_manager = TempFileManager()
    avi_temp = temp_manager.create_temp_file('.avi').name
    final_avi_temp = temp_manager.create_temp_file('.avi').name
    out_video = cv2.VideoWriter(avi_temp, cv2.VideoWriter_fourcc(*'XVID'), 25, (224,224))
    dets = {'x':[], 'y':[], 's':[], 'x1':[], 'y1':[], 'w':[], 'h':[], 'face':[] }
    for det in track['bbox']: # Read the tracks
        dets['s'].append(max(det[2], det[3]) / 2)
        dets['x'].append(det[0] + det[2] / 2)
        dets['y'].append(det[1] + det[3] / 2)
        dets['x1'].append(int(det[0]))
        dets['y1'].append(int(det[1]))
        dets['w'].append(int(det[2]))
        dets['h'].append(int(det[3]))
    
    dets['s'] = signal.medfilt(dets['s'], kernel_size=13)  # Smooth detections 
    dets['x'] = signal.medfilt(dets['x'], kernel_size=13)
    dets['y'] = signal.medfilt(dets['y'], kernel_size=13)

    for fidx, frame in enumerate(track['frame']):
        cs = crop_scale
        bs = dets['s'][fidx] # Detection box size
        bsi = int(bs * (1 + 2 * cs)) # Pad videos by this amount 
        padded_frame = np.pad(image[frame], ((bsi, bsi), (bsi, bsi), (0, 0)), 'constant', constant_values=(110, 110))
        my = dets['y'][fidx] + bsi # BBox center Y
        mx = dets['x'][fidx] + bsi # BBox center X
        face = padded_frame[int(my-bs):int(my+bs*(1+2*cs)), int(mx-bs*(1+cs)):int(mx+bs*(1+cs))]
        out_video.write(cv2.resize(face, (224, 224)))
        # Storing original face
        face = image[frame][dets['y1'][fidx]:dets['y1'][fidx]+dets['h'][fidx], \
                    dets['x1'][fidx]:dets['x1'][fidx]+dets['w'][fidx]]
        dets['face'].append(face)
    
    audio_tmp = temp_manager.create_temp_file('.wav').name
    audio_start = (track['frame'][0]) / 25
    audio_end = (track['frame'][-1]+1) / 25

    out_video.release()
    
    command = ("ffmpeg -y -i %s -async 1 -ac 1 -vn -acodec pcm_s16le -ar 16000 -threads %d -ss %.3f -to %.3f %s -loglevel panic" % \
            (audio_file_path, loader_threads, audio_start, audio_end, audio_tmp)) 
    subprocess.call(command, shell=True, stdout=None) # Crop audio file
    command = ("ffmpeg -y -i %s -i %s -threads %d -c:v copy -c:a copy %s -loglevel panic" % \
            (avi_temp, audio_tmp, loader_threads, final_avi_temp)) # Combine audio and video file
    subprocess.call(command, shell=True, stdout=None)

    return {'track': track, 'proc_track': dets }, final_avi_temp, audio_tmp

def evaluate_network(videos, audios):
    if not os.path.exists('weights/pretrain_TalkSet.model'):
        raise Exception('No weights. Download it first')
    s = talkNet()
    s.loadParameters('weights/pretrain_TalkSet.model')
    all_scores = []
    durationSet = {1,1,1,2,2,2,3,3,4,5,6} # Use this line can get more reliable result

    for i, video in enumerate(videos):
        _, audio = wavfile.read(audios[i])
        audio_features = python_speech_features.mfcc(audio, 16000, numcep = 13, winlen = 0.025, winstep = 0.010)
        video = cv2.VideoCapture(video)
        video_features = []

        while video.isOpened():
            ret, frames = video.read()
            if ret == True:
                face = cv2.cvtColor(frames, cv2.COLOR_BGR2GRAY)
                face = cv2.resize(face, (224,224))
                face = face[int(112-(112/2)):int(112+(112/2)), int(112-(112/2)):int(112+(112/2))]
                video_features.append(face)
            else:
                break

        video.release()
        video_features = np.array(video_features)

        length = min((audio_features.shape[0] - audio_features.shape[0] % 4) / 100, video_features.shape[0] / 25)
        audio_features = audio_features[:int(round(length * 100)),:]
        videoFeature = video_features[:int(round(length * 25)),:,:]
        all_score = [] # Evaluation use TalkNet

        for duration in durationSet:
            batchSize = int(math.ceil(length / duration))
            scores = []
            with torch.no_grad():
                for j in range(batchSize):
                    inputA = torch.FloatTensor(audio_features[j * duration * 100:(j+1) * duration * 100,:]).unsqueeze(0).cpu()
                    inputV = torch.FloatTensor(videoFeature[j * duration * 25: (j+1) * duration * 25,:,:]).unsqueeze(0).cpu()
                    if inputA.size(1) >= 4 and inputV.size(1) > 1:
                        embedA = s.model.forward_audio_frontend(inputA)
                        embedV = s.model.forward_visual_frontend(inputV)
                        embedA, embedV = s.model.forward_cross_attention(embedA, embedV)
                        out = s.model.forward_audio_visual_backend(embedA, embedV)
                        score = s.lossAV.forward(out, labels = None)
                        scores.extend(score)
            all_score.append(scores)
        all_score = np.round((np.mean(np.array(all_score), axis = 0)), 1).astype(float)
        all_scores.append(all_score)	
    return all_scores

def map_scores(scores, tracks):
    outputs = dict()
    for tidx, track in enumerate(tracks):
        score = scores[tidx]
        for fidx, frame in enumerate(track['track']['frame'].tolist()):
            s = score[max(fidx - 2, 0): min(fidx + 3, len(score) - 1)] # average smoothing
            s = np.mean(s)
            outputs[frame] = {
                'track':tidx, 'score':float(s),
                'face': track['proc_track']['face'][fidx],
                'bbox': [
                    track['proc_track']['x1'][fidx],
                    track['proc_track']['y1'][fidx],
                    track['proc_track']['w'][fidx],
                    track['proc_track']['h'][fidx]
                ]
            }

    return outputs

def detect_scores(config, video_file_path):
    scene = detect_scene(video_file_path)
    faces, frames, audio_temp_file = find_faces(video_file_path, config)
    allTracks, vidTracks, videos, wavs = [], [], [], []
    for shot in scene:
        allTracks.extend(track_shot(faces[shot[0].frame_num:shot[1].frame_num]))
    for _, track in tqdm(enumerate(allTracks), total = len(allTracks)):
        vidTrack, avi, wav = crop_video(track, audio_temp_file, frames)
        vidTracks.append(vidTrack)
        videos.append(avi)
        wavs.append(wav)
    scores = evaluate_network(videos, wavs)
    return map_scores(scores, vidTracks)