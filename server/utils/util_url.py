import re

from server.config import IMG_FORMATS


def get_name_image_from_url(img_url):
    img_format = '|'.join(IMG_FORMATS)
    matches = re.findall(f'(.*)\/(.*?)\.({img_format})(.*)', img_url)
    return list(map(lambda x: f"{x[1]}.{x[2]}", matches))


if __name__ == '__main__':
    _url = 'https://storage.googleapis.com/vucar-user-assets/599a101f-d78b-4756-80af-9060e01c27ec_49088482-db67-44c6' \
           '-8d54-2e5b934bf4e9_1684214364144_z4296840083462_ee93067db80aa9ee256d67a23dd8c198.webp?alsdflashdf'
    print(get_name_image_from_url(_url))
