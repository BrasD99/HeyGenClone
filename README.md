<h1 align="center">HeyGenClone</h1>

<p>
  Welcome to <strong>HeyGenClone</strong>, an open-source analogue of the HeyGen system.
</p>

<p>
  I am a developer from Moscow 🇷🇺 who devotes his free time to studying new technologies. The project is in an active development phase, but I hope it will help you achieve your goals!
</p>

<p>
  Currently, translation support is enabled only from English 🇬🇧!
</p>

<p align="center">
  <img src="https://i.ibb.co/N2w50HD/corgi.jpg" width="100%" height="auto" />
</p>

## Installation
- Clone this repo
- Install [conda](https://conda.io/projects/conda/en/latest/user-guide/install/)
- Create environment and install requirements:
  ```
  cd path_to_project
  sh install/env.sh
  ```
- In config.json file change HF_TOKEN argument. It is your HuggingFace token. Visit [speaker-diarization](https://hf.co/pyannote/speaker-diarization), [segmentation](https://hf.co/pyannote/segmentation) and accept user conditions
- Download weights from [drive](https://drive.google.com/file/d/1dYy24q_67TmVuv_PbChe2t1zpNYJci1J/view?usp=sharing), unzip downloaded file into <strong>weights</strong> folder
- Install [ffmpeg](https://ffmpeg.org/)

## Configurations (config.json)
| Key | Description | Can modify |
|     :---:      |     :---:     |     :---:      |
|     LANGUAGES_URL      |     Url for getting available languages     |     ❌      |
|     DET_TRESH      |     Face detection treshtold [0.0:1.0]     |     ✅      |
|     DIST_TRESH      |     Face embeddings distance treshtold [0.0:1.0]     |     ✅      |
|     DB_NAME      |     Name of the database for data storage     |     ✅      |
|     HF_TOKEN      |     Your HuggingFace token (see [Installation](https://github.com/BrasD99/HeyGenClone/tree/main#installation))     |     ✅      |

## Usage
```
  conda activate heygenclone
  cd path_to_project
```
At the root of the project there is a translate script that translates the video you set.
- video_filename - the filename of your input video (.mp4)
- output_language - the code of the language to be translated into (you can find it [here](https://dl.fbaipublicfiles.com/mms/tts/all-tts-languages.html))
- output_filename - the filename of output video (.mp4)
```
python translate.py video_filename output_language -o output_filename
```

I also added a script to overlay the voice on the video with lip sync, which allows you to create a video with a person pronouncing your speech. Сurrently it works for videos with one person.
- voice_filename - the filename of your speech (.wav)
- video_filename - the filename of your input video (.mp4)
- output_filename - the filename of output video (.mp4)
```
python speech_changer.py voice_filename video_filename -o output_filename
```

## How it works
1. Detecting scenes ([PySceneDetect](https://github.com/Breakthrough/PySceneDetect))
2. Face detection ([yolov8-face](https://github.com/akanametov/yolov8-face))
3. Reidentification ([deepface](https://github.com/serengil/deepface))
4. Speech enhancement ([MDXNet](https://huggingface.co/freyza/kopirekcover/blob/main/MDXNet.py))
5. Speakers transcriptions and diarization ([whisperX](https://github.com/m-bain/whisperX))
6. Text translation ([googletrans](https://pypi.org/project/googletrans/))
7. Voice cloning ([TTS](https://github.com/coqui-ai/TTS))
8. Lip sync ([lipsync](https://github.com/mowshon/lipsync))
9. Face restoration ([GFPGAN](https://github.com/TencentARC/GFPGAN))
10. [Need to fix] Search for talking faces, determining what this person is saying

## Translation results
Note that this example was created without GFPGAN usage!
| Destination language | Source video | Output video |
|     :---:      |     :---:     |     :---:      |
|🇷🇺 (Russian)     | [![Watch the video](https://i.ibb.co/KD2KKnj/en.jpg)](https://youtu.be/eGFLPAQAC2Y)    | [![Watch the video](https://i.ibb.co/cbwCy8F/ru.jpg)](https://youtu.be/L2YTmfIr7aI)    |

## To-Do List
- [ ] Fully GPU support
- [ ] Multithreading support (optimizations)
- [ ] Detecting talking faces (improvement)

## Other
- Tested on macOS
- :warning: The project is under development!
