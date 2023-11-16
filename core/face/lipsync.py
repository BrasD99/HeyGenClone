import torch
from .models.wav2lip import Wav2Lip
from .audio import load_wav, melspectrogram
from core.gfpganer import GFPGANer
import numpy as np
import cv2
import os
import tempfile


class LipSync:
    def __init__(self):
        checkpoint_path = os.path.join('weights', 'wav2lip_gan.pth')
        if not os.path.exists(checkpoint_path):
            raise Exception('Download weights!')
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = self.load_model(checkpoint_path)

        # If True, then use only first video frame for inference.
        self.static = False
        # Can be specified only if input is a static image (default: 25).
        self.fps = 25.
        # Padding (top, bottom, left, right). Please adjust to include chin at least.
        self.pads = [0, 10, 0, 0]
        # Batch size for face detection.
        self.face_det_batch_size = 16
        # Batch size for Wav2Lip model(s).
        self.wav2lip_batch_size = 128
        # Reduce the resolution by this factor. Sometimes, best results are obtained at 480p or 720p.
        self.resize_factor = 1
        # Crop video to a smaller region (top, bottom, left, right). Applied after resize_factor and rotate arg. '
        # 'Useful if multiple face present. -1 implies the value will be auto-inferred based on height, width
        self.crop = [0, -1, 0, -1]
        # Specify a constant bounding box for the face. Use only as a last resort if the face is not detected.'
        # 'Also, might work only if the face is not moving around much. Syntax: (top, bottom, left, right).
        self.box = [-1, -1, -1, -1]
        # Sometimes videos taken from a phone can be flipped 90deg. If true, will flip video right by 90deg.'
        # 'Use if you get a flipped result, despite feeding a normal looking video
        self.rotate = False
        # Prevent smoothing face detections over a short temporal window
        self.nosmooth = False
        # This option enables caching for face positioning on every frame.
        # Useful if you will be using the same video, but different audio.
        self.save_cache = True
        self.cache_dir = tempfile.gettempdir()
        self.img_size = 96
        self.mel_step_size = 16
        self.box = [-1, -1, -1, -1]
        device_type = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.restorer = GFPGANer(
            model_path=os.path.join('weights', 'GFPGANv1.4.pth'),
            root_dir='weights',
            upscale=2,
            arch='clean',
            channel_multiplier=2,
            bg_upsampler=None,
            device=device_type
        )

    def load_model(self, checkpoint_path):
        model = Wav2Lip()
        # print('Load checkpoint from: {}'.format(checkpoint_path))
        checkpoint = self._load(checkpoint_path)
        s = checkpoint['state_dict']
        new_s = {}
        for k, v in s.items():
            new_s[k.replace('module.', '')] = v
        model.load_state_dict(new_s)

        model = model.to(self.device)
        return model.eval()

    def _load(self, checkpoint_path):
        if self.device == 'cuda':
            checkpoint = torch.load(checkpoint_path)
        else:
            checkpoint = torch.load(checkpoint_path,
                                    map_location=lambda storage, loc: storage)
        return checkpoint

    def datagen(self, frames_dict, mels):
        img_batch, mel_batch, frame_batch, coords_batch, frame_ids = [], [], [], [], []

        for i, m in enumerate(mels):
            idx = 0 if self.static else i % len(frames_dict)

            frame_to_save = frames_dict[idx]['frame'].copy()

            if frames_dict[idx]['has_face']:
                face = frames_dict[idx]['face'].copy()
                coords = frames_dict[idx]['bbox']
                face = cv2.resize(face, (self.img_size, self.img_size))
                img_batch.append(face)
                coords_batch.append(coords)
                mel_batch.append(m)
                frame_batch.append(frame_to_save)
                frame_ids.append(idx)

            if len(img_batch) >= self.wav2lip_batch_size:
                img_batch, mel_batch = np.asarray(
                    img_batch), np.asarray(mel_batch)

                img_masked = img_batch.copy()
                img_masked[:, self.img_size // 2:] = 0

                img_batch = np.concatenate(
                    (img_masked, img_batch), axis=3) / 255.
                mel_batch = np.reshape(
                    mel_batch, [len(mel_batch), mel_batch.shape[1], mel_batch.shape[2], 1])

                yield img_batch, mel_batch, frame_batch, coords_batch, frame_ids
                img_batch, mel_batch, frame_batch, coords_batch, frame_ids = [], [], [], [], []

        if len(img_batch) > 0:
            img_batch, mel_batch = np.asarray(img_batch), np.asarray(mel_batch)

            img_masked = img_batch.copy()
            img_masked[:, self.img_size // 2:] = 0

            img_batch = np.concatenate((img_masked, img_batch), axis=3) / 255.
            mel_batch = np.reshape(
                mel_batch, [len(mel_batch), mel_batch.shape[1], mel_batch.shape[2], 1])

            yield img_batch, mel_batch, frame_batch, coords_batch, frame_ids

    def sync(self, frames_dict, audio_file, fps, use_enhancer):
        wav = load_wav(audio_file, 16000)
        mel = melspectrogram(wav)
        # print(mel.shape)

        if np.isnan(mel.reshape(-1)).sum() > 0:
            raise ValueError(
                'Mel contains nan! Using a TTS voice? Add a small epsilon noise to the wav file and try again')

        mel_chunks = []
        mel_idx_multiplier = 80. / fps

        i = 0
        while 1:
            start_idx = int(i * mel_idx_multiplier)
            if start_idx + self.mel_step_size > len(mel[0]):
                mel_chunks.append(mel[:, len(mel[0]) - self.mel_step_size:])
                break
            mel_chunks.append(
                mel[:, start_idx: start_idx + self.mel_step_size])
            i += 1

        # print('Length of mel chunks: {}'.format(len(mel_chunks)))

        gen = self.datagen(frames_dict, mel_chunks)

        for i, (img_batch, mel_batch, frames, coords, frame_ids) in \
                enumerate(gen):

            img_batch = torch.FloatTensor(np.transpose(
                img_batch, (0, 3, 1, 2))).to(self.device)
            mel_batch = torch.FloatTensor(np.transpose(
                mel_batch, (0, 3, 1, 2))).to(self.device)

            with torch.no_grad():
                pred = self.model(mel_batch, img_batch)

            pred = pred.cuda() if self.device == 'cuda' else pred.cpu()
            pred = pred.cpu().numpy().transpose(0, 2, 3, 1) * 255.

            for p, f, c, i in zip(pred, frames, coords, frame_ids):
                [x1, y1, w, h] = c
                face = cv2.resize(p.astype(np.uint8), (w, h))
                if use_enhancer:
                    _, _, r_img = self.restorer.enhance(face)
                    f[y1:y1+h, x1:x1+w] = cv2.resize(r_img, (w, h))
                else:
                    f[y1:y1+h, x1:x1+w] = face
                frames_dict[i]['frame'] = f

        return frames_dict
