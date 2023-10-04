<h1 align="center">HeyGenClone</h1>

<p>
  Welcome to <strong>HeyGenClone</strong>, an open-source analogue of the HeyGen system.
</p>

<p>
  I am a developer from Moscow 🇷🇺 who devotes his free time to studying new technologies. The project is in an active development phase, but I hope it will help you achieve your goals!
</p>

## Installation
- Clone this repo
- In config.json file change HF_TOKEN argument. It is your HuggingFace token. Visit https://hf.co/pyannote/speaker-diarization, https://hf.co/pyannote/segmentation and accept user conditions
- Download weights from https://drive.google.com/file/d/1e35OvOlWVNndkx0Gv7zc5emwnX7t3Oc4/view?usp=sharing, unzip downloaded file into <strong>weights</strong> folder

## Usage
At the root of the project there is a translate script that translates the movie you set.
```
python translate.py video_filename output_language -o output_filename
```

## Conversion results
| Src lang (detected) | Dst lang | Src video | Final video |
|     :---:      |     :---:      |     :---:     |     :---:      |
| 🇬🇧   | 🇷🇺     | [![Watch the video](https://i.ibb.co/KD2KKnj/en.jpg)](https://youtu.be/eGFLPAQAC2Y)    | [![Watch the video](https://i.ibb.co/cbwCy8F/ru.jpg)](https://youtu.be/L2YTmfIr7aI)    |



