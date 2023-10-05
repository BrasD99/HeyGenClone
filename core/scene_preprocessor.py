from typing import Any
from scenedetect.video_manager import VideoManager
from scenedetect.scene_manager import SceneManager
from scenedetect.stats_manager import StatsManager
from scenedetect.detectors import ContentDetector
from core.face.detector import FaceDetector
from core.temp_manager import TempFileManager
from core.dereverb import MDXNetDereverb
from deepface import DeepFace
from deepface.commons import distance as dst
from pydub import AudioSegment
import sqlite3
import uuid
import pickle
import os
import shutil
import numpy as np
from tqdm import tqdm

class ScenePreprocessor:
    def __init__(self, config):
        self.face_detector = FaceDetector()
        self.dereverb = MDXNetDereverb(15)
        self.temp_manager = TempFileManager()
        self.dist_tresh = config['DIST_TRESH']
        self.face_det_tresh = config['DET_TRESH']
        self.conn = self.create_db(config['DB_NAME'])

    def __call__(self, clip, video_file_path, voice_segments):
        scenes = self.detect_scenes(video_file_path)

        frame_id = 0

        for scene_id, scene in enumerate(scenes):
            voice_frame_ids = []
            start, end = scene
            subclip = clip.subclip(start.frame_num / clip.fps, end.frame_num / clip.fps)
            for frame in tqdm(subclip.iter_frames(), desc=f'Face detector [scene_id: {scene_id + 1}]'):
                self.insert_frame(frame_id, frame)
                frame_time = frame_id / clip.fps
                if self.is_frame_with_voice(frame_time, voice_segments):
                    voice_frame_ids.append(frame_id)
                    faces = self.face_detector.detect(frame, face_det_tresh=self.face_det_tresh)
                    for face in faces:
                        embedding = DeepFace.represent(face[0], enforce_detection=False)[0]['embedding']
                        self.find_insert_embedding(embedding, frame_id, face[0], face[1])
                frame_id += 1

    def is_frame_with_voice(self, frame_time, voice_segments):
        for voice_segment in voice_segments:
            if frame_time >= voice_segment[0] and frame_time <= voice_segment[1]:
                return True
        return False
    
    def to_pydub_audio(self, clip_audio):
        temp_manager = TempFileManager()
        temp_file = temp_manager.create_temp_file(suffix='.wav').name
        clip_audio.write_audiofile(temp_file, codec='pcm_s16le')
        return AudioSegment.from_file(temp_file, format='wav'), temp_file
    
    def detect_scenes(self, video_file_path):
        videoManager = VideoManager([video_file_path])
        statsManager = StatsManager()
        sceneManager = SceneManager(statsManager)
        sceneManager.add_detector(ContentDetector())
        baseTimecode = videoManager.get_base_timecode()
        videoManager.set_downscale_factor()
        videoManager.start()
        sceneManager.detect_scenes(frame_source = videoManager)
        return sceneManager.get_scene_list(baseTimecode, start_in_scene=True)
    
    def close(self):
        self.conn.close()
    
    def get_persons_on_frame(self, frame_id):
        cursor = self.conn.execute('SELECT person_id FROM embeddings WHERE frame_id=?', (frame_id,))
        rows = cursor.fetchall()
        output = []
        for row in rows:
            output.append(row[0])
        return output
    
    def create_db(self, db_name):
        if os.path.exists(db_name):
            os.remove(db_name)
        
        conn = sqlite3.connect(db_name)

        conn.execute('''CREATE TABLE IF NOT EXISTS persons
                (person_id TEXT PRIMARY KEY)''')
        conn.execute('''CREATE TABLE IF NOT EXISTS embeddings
                    (embedding_id INTEGER PRIMARY KEY AUTOINCREMENT, person_id TEXT, embedding BLOB, frame_id INTEGER, face BLOB, bbox BLOB,
                    FOREIGN KEY(person_id) REFERENCES persons(person_id))''')
        conn.execute('''CREATE TABLE IF NOT EXISTS frames
                (frame_id INTEGER PRIMARY KEY, frame BLOB)''')
        return conn
    
    def get_face_on_frame(self, person_id, frame_id):
        cursor = self.conn.execute('SELECT face, bbox FROM embeddings WHERE person_id=? AND frame_id=?', (person_id, frame_id))
        row = cursor.fetchone()
        if row:
            return {
                'face': np.array(pickle.loads(row[0])),
                'bbox': np.array(pickle.loads(row[1]))
            }

        return None
    
    def insert_frame(self, frame_id, frame):
        frame_bytes = pickle.dumps(frame)
        self.conn.execute('INSERT INTO frames (frame_id, frame) VALUES (?, ?)', (frame_id, sqlite3.Binary(frame_bytes)))
        self.conn.commit()

    def get_frames(self):
        frames = dict()
        cursor = self.conn.execute('SELECT * FROM frames')
        rows = cursor.fetchall()
        for row in rows:
            frames[row[0]] = pickle.loads(row[1])
        return frames
    
    def find_insert_embedding(self, embedding, frame_id, face, bbox):
        embeddings_dict = self.get_all_persons_with_embeddings()
        for person_id, embeddings in embeddings_dict.items():
            for person_embedding in embeddings:
                distance = dst.findEuclideanDistance(embedding, person_embedding)
                if distance <= self.dist_tresh:
                    self.insert_person_embedding(person_id, embedding, frame_id, face, bbox)
                    return
        
        person_id = self.generate_new_person_id()
        self.insert_embedding(person_id, embedding, frame_id, face, bbox)

    def get_all_persons_with_embeddings(self):
        persons = self.get_all_persons()
        persons_with_embeddings = {}
        for person_id in persons:
            embeddings = self.get_embeddings(person_id)
            persons_with_embeddings[person_id] = embeddings
        return persons_with_embeddings
    
    def get_all_persons(self):
        cursor = self.conn.execute('SELECT person_id FROM persons')
        rows = cursor.fetchall()
        persons = [row[0] for row in rows]
        return persons
    
    def get_embeddings(self, person_id):
        cursor = self.conn.execute('SELECT embedding FROM embeddings WHERE person_id=?', (person_id,))
        rows = cursor.fetchall()
        embeddings = []
        for row in rows:
            embeddings.append(pickle.loads(row[0]))
        return embeddings
    
    def insert_embedding(self, person_id, embedding, frame_id, face, bbox):
        embedding_bytes = pickle.dumps(embedding)
        face_bytes = pickle.dumps(face)
        bbox_bytes = pickle.dumps(bbox)
        self.conn.execute('INSERT INTO persons (person_id) VALUES (?)', (person_id,))
        self.conn.execute('INSERT INTO embeddings (person_id, embedding, frame_id, face, bbox) VALUES (?, ?, ?, ?, ?)',
                    (person_id, sqlite3.Binary(embedding_bytes), frame_id, sqlite3.Binary(face_bytes), sqlite3.Binary(bbox_bytes)))
        self.conn.commit()

    def insert_person_embedding(self, person_id, embedding, frame_id, face, bbox):
        embedding_bytes = pickle.dumps(embedding)
        face_bytes = pickle.dumps(face)
        bbox_bytes = pickle.dumps(bbox)
        self.conn.execute('INSERT INTO embeddings (person_id, embedding, frame_id, face, bbox) VALUES (?, ?, ?, ?, ?)',
                    (person_id, sqlite3.Binary(embedding_bytes), frame_id, sqlite3.Binary(face_bytes), sqlite3.Binary(bbox_bytes)))
        self.conn.commit()

    def generate_new_person_id(self):
        return str(uuid.uuid4())
    
    def get_all_persons_with_embeddings(self):
        persons = self.get_all_persons()
        persons_with_embeddings = {}
        for person_id in persons:
            embeddings = self.get_embeddings(person_id)
            persons_with_embeddings[person_id] = embeddings
        return persons_with_embeddings