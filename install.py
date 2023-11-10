import subprocess

requirements = {
    'TTS': 'TTS==0.17.6',
    'pyannote.core': 'pyannote.core==5.0.0',
    'pyannote.database': 'pyannote.database==5.0.1',
    'pyannote.pipeline': 'pyannote.pipeline==3.0.1',
    'torchaudio': 'torchaudio==2.0.2',
    'audiostretchy': 'audiostretchy==1.3.5',
    'beautifulsoup4': 'beautifulsoup4==4.12.2',
    'ctranslate2': 'ctranslate2==3.20.0',
    'deepface': 'deepface==0.0.79',
    'faster_whisper': 'faster_whisper==0.9.0',
    'ffmpeg_python': 'ffmpeg_python==0.2.0',
    'googletrans': 'googletrans==3.1.0a0',
    'librosa': 'librosa==0.8.1',
    'lws': 'lws==1.2.7',
    'moviepy': 'moviepy==1.0.3',
    'nltk': 'nltk==3.8.1',
    'onnxruntime': 'onnxruntime==1.16.0',
    'opencv_python': 'opencv_python==4.7.0.68',
    'opencv_python_headless': 'opencv_python_headless==4.7.0.68',
    'pandas': 'pandas==1.5.3',
    'pyannote.audio': 'pyannote.audio==2.1.1',
    'pyannote.metrics': 'pyannote.metrics==3.2.1',
    'pydub': 'pydub==0.25.1',
    'Requests': 'Requests==2.31.0',
    'scenedetect': 'scenedetect==0.6.2',
    'scipy': 'scipy==1.11.3',
    'torch': 'torch==2.0.1',
    'tqdm': 'tqdm==4.65.0',
    'transformers': 'transformers==4.33.2',
    'ultralytics': 'ultralytics==8.0.147',
    'chardet': 'chardet',
    'basicsr': 'basicsr',
    'facexlib': 'facexlib',
    'gfpgan': 'gfpgan',
    'pyannote.audio [fix]': 'pyannote.audio',
    'torch & torchaudio [fix]': '-U torch torchaudio --no-cache-dir',
    'numpy [fix]': 'numpy==1.23'
}

for name, requirement in requirements.items():
    print(f'Installing package "{name}"...')
    subprocess.run(
        f'pip install {requirement}',
        shell=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL)

print('Done!')
