from googletrans import Translator

class TextHelper:
    def __init__(self):
        self.translator = Translator()

    def translate(self, text, src_lang, dst_lang):
        print(text, ' ', src_lang, ' ', dst_lang)
        output = self.translator.translate(text, src=src_lang, dest=dst_lang)
        return output.text