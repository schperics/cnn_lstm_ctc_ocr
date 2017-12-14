from PIL import Image
from PIL import ImageFont
from PIL import ImageDraw
import numpy as np
import random
import re
import tensorflow as tf


def load_word():
    words = []
    alphabet = {}
    alphabet[' '] = 0
    special_chars = re.compile('[\.,"?”“-…()‘’!\'-子拍陸]')
    for l in open("sample.txt", "rb"):
        for w in str(l, "utf-8").split():
            w = special_chars.sub('', w).strip()
            if len(w) == 0:
                continue
            words.append(w)
            for a in w:
                alphabet[a] = 0
    alphabet = sorted(alphabet.keys())
    return words, alphabet


def synth_image(text, height, scale, padding=1):
    while True:
        background = random.randint(0, 255)
        foreground = random.randint(0, 255)
        if abs(background - foreground) > 100:
            break

    font_size = int((height - 2 * padding) * scale)
    font = ImageFont.truetype("/mnt/SDMiSaeng.ttf", font_size)
    width = font.getmask(text).size[0] + 2 * padding
    image = Image.fromarray(np.ones((height, width), np.int8) * background, "L")
    draw = ImageDraw.Draw(image)
    draw.text((padding, padding), text, (foreground), font=font)
    return image


def dense_to_sparse(t, length):
    idx = tf.where(tf.not_equal(t, 0))
    sparse = tf.SparseTensor(idx, tf.gather_nd(t, idx), [length])
    return sparse


class ToySynth(object):
    def __init__(self, include_none=True):
        words, alphabets = load_word()
        self.words = words
        self.alphabets = alphabets
        self.reverse_index = {}
        self.include_none = include_none

        index = 0
        if include_none:
            self.reverse_index[0] = None
            index += 1

        for c in alphabets:
            self.reverse_index[c] = index
            index += 1

    def num_classes(self):
        return len(self.alphabets) + (1 if self.include_none else 0)

    def random_words(self, max_words=3):
        c = random.randint(1, max_words)
        text = ''
        for i in range(c):
            if i > 0:
                text += ' '
            i = random.randint(0, len(self.words) - 1)
            text += self.words[i]
        index = [self.reverse_index[c] for c in text]
        return text, index

    def label_to_text(self, label):
        text = ''
        for i in label:
            if self.include_none:
                if i == 0:
                    continue
                text += self.alphabets[i - 1]
            else:
                text += self.alphabets[i]
        return text

    def _random_image(self, height=32, scale=1.0, padding=1, max_words=3):
        text, label = self.random_words(max_words)
        image = synth_image(text, height=height, scale=scale, padding=padding)
        width, height = image.size
        # image.save("bar.jpg")
        # print(text)
        return np.expand_dims(np.array(image), axis=-1), width, label, len(label)

    def get(self):
        def generator():
            while True:
                yield self._random_image()

        dataset = tf.data.Dataset.from_generator(
            generator,
            (tf.uint8, tf.int32, tf.int64, tf.int64),
            (tf.TensorShape([None, None, 1]),  # image
             tf.TensorShape([]),  # width
             tf.TensorShape([None]),  # label
             tf.TensorShape([])))  # length
        image, width, label, length = dataset.make_one_shot_iterator().get_next()
        label = tf.serialize_sparse(dense_to_sparse(label, length))
        return image, width, label, length


def _test():
    t = ToySynth()
    text, label = t.random_words()
    print(t.num_classes())
    print(text)
    print(t.label_to_text(label))

    """
    d = t.get()
    with tf.Session() as sess:
        image, width, label, length = sess.run(d)
        print("image : ", image.shape)
        print("width : ", width)
        print("label : ", label)
        print("length : ", length)
    """


if __name__ == "__main__":
    _test()
