DEFAULT_VIDEO_LANGS = ['en']


mapper = {
    'english': 'en',
    'spanish': 'es',
    'french': 'fr',
    'german': 'de',
    'italian': 'it',
    'portuguese': 'pt',
    'polish': 'pl',
    'turkish': 'tr',
    'russian': 'ru',
    'dutch': 'nl',
    'czech': 'cs',
    'arabic': 'ar',
    'chinese': 'zh-cn',
    'japanese': 'ja',
    'hungarian': 'hu',
    'korean': 'ko'
}


def get_languages():
    return list(mapper.keys())


def is_valid_lang(language):
    languages = get_languages()
    return language.lower() in languages

def map(language):
    return mapper[language]
