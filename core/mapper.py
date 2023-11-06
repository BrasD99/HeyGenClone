mapper = {
    'en': 'en',
    'rus': 'ru',
    'zh': 'zh-CN'
}

def map(code):
    if not code in mapper:
        raise Exception(f'Language {code} is not currently supported! Please write me at https://t.me/+IlOPXyNkscxhZjJi')
    return mapper[code]