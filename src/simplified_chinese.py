import os
import codecs
import glob
import numpy as np
from PIL import Image
from PIL import ImageFont
from PIL import ImageDraw

_additional_chars = ' 0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ'
_base_dir = "/mnt/chinese"
_alphabet_file = "gb2312.txt"
_text_files = "text/conv_*"
_word_file = "words.txt"
_sample_words = "10k_words.txt"


def _make_alphabet():
    alphabet = []
    alphabet.append('')
    for c in _additional_chars:
        alphabet.append(c)
    for l in codecs.open(os.path.join(_base_dir, _alphabet_file), "r", "utf-8"):
        alphabet.append(l.strip())

    rindex = {}
    for i, c in enumerate(alphabet):
        rindex[c] = i

    return alphabet, rindex


def _word_is_not_chinese(word):
    for c in word:
        if c in alphabet and c not in _additional_chars:
            return False
    return True


def make_words():
    text_files = glob.glob(os.path.join(_base_dir, _text_files))
    with codecs.open(os.path.join(_base_dir, _word_file), "w", "utf-8") as word_file:
        for text_file in text_files:
            for l in codecs.open(text_file, "r", "utf-8"):
                w = ''
                for c in l:
                    if c in rindex and c != ' ':
                        w += c
                    else:
                        w = w.strip()
                        if w == '':
                            continue
                        if not _word_is_not_chinese(w):
                            while len(w) > _max_length:
                                s = w[0:_max_length]
                                w = w[_max_length:]
                                word_file.write(s + '\n')
                            word_file.write(w + '\n')
                        w = ''


alphabet, rindex = _make_alphabet()
_max_length = 10
words = [w.strip() for w in codecs.open(os.path.join(_base_dir, _sample_words), "r", "utf-8")]
fonts = glob.glob(os.path.join(_base_dir, "fonts/*.ttf"))


def _fonts_sample():
    height = len(fonts) * 32 + 6
    image = Image.fromarray(np.ones((height, 600), np.int8) * 0, "L")
    draw = ImageDraw.Draw(image)
    for i, font_path in enumerate(fonts):
        font_name = os.path.basename(font_path)
        font = ImageFont.truetype(font_path, 30)
        draw.text((3, 3 + 32 * i), font_name + " : 中国字体", (255), font=font)
    image.save(os.path.join(_base_dir, "font_sample.jpg"))


if __name__ == "__main__":
    _fonts_sample()
