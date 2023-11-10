from googletrans import Translator
from core.mapper import map_to_trans


class TextHelper:
    def __init__(self):
        self.translator = Translator()

    def translate(self, text, src_lang, dst_lang):
        output = self.translator.translate(
            text, src=src_lang, dest=map_to_trans(dst_lang))
        return output.text
