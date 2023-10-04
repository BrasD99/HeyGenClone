import argparse
import requests
import json
from bs4 import BeautifulSoup
from core.engine import Engine

with open('config.json', 'r') as f:
    config = json.load(f)

LANGUAGES_URL = config['LANGUAGES_URL']

def contains_only_ascii(input_string):
    return all(ord(char) < 128 for char in input_string)

def get_iso_languages():
    response = requests.get(LANGUAGES_URL)
    soup = BeautifulSoup(response.text, 'html.parser')

    p_tags = soup.find_all('p')

    iso_language_dict = {}

    for p_tag in p_tags[1:]:  # Skipping the first <p> which contains the header
        parts = p_tag.get_text().split()
        if len(parts) == 2:
            iso_code, language_name = parts
            if contains_only_ascii(language_name):
                iso_language_dict[language_name] = iso_code

    return iso_language_dict

def translate(video_filename, output_language, output_filename):
    engine = Engine(config, output_language)
    engine(video_filename, output_filename)

if __name__ == '__main__':
    langs = get_iso_languages()
    parser = argparse.ArgumentParser(description='Combine an audio file and a video file into a new video file')
    parser.add_argument('video_filename', help='path to video file')
    parser.add_argument('output_language', choices=list(langs.values()), default='rus', help='choose one option')
    parser.add_argument('-o', '--output_filename', default='output.mp4', help='output file name (default: output.mp4)')
    args = parser.parse_args()

    translate(
        video_filename=args.video_filename,
        output_language=args.output_language,
        output_filename=args.output_filename,
    )