import numpy as np
import random
import tensorflow as tf
import Hangulpy as han
import data_util


class ToySynth(object):
    def __init__(self):
        self.words, alphabet = data_util.load_word()

        self.fidx = {}
        self.midx = {}
        self.lidx = {}
        self.rfidx = {}
        self.rmidx = {}
        self.rlidx = {}

        def _ins(v, idx, ridx):
            if v not in ridx:
                i = len(idx)
                idx[i] = v
                ridx[v] = i

        for c in alphabet:
            if c == ' ':
                continue
            f, m, l = han.decompose(c)
            _ins(f, self.fidx, self.rfidx)
            _ins(m, self.midx, self.rmidx)
            _ins(l, self.lidx, self.rlidx)

        self.length = (len(self.fidx), len(self.midx), len(self.lidx))

    def _char_to_index(self, c):
        f, m, l = han.decompose(c)
        f = self.rfidx[f]
        m = self.rmidx[m]
        l = self.rlidx[l]
        return l + self.length[2] * (m + f * self.length[1]) + 1

    def _index_to_char(self, idx):
        if idx == 0:
            return ''
        idx -= 1
        l = idx % self.length[2]
        idx = idx // self.length[2]
        m = idx % self.length[1]
        f = idx // self.length[1]
        f = self.fidx[f]
        m = self.midx[m]
        l = self.lidx[l]
        return han.compose(f, m, l)

    def random_words(self, max_words=3):
        c = random.randint(2, max_words)
        text = ''
        for i in range(c):
            i = random.randint(0, len(self.words) - 1)
            text += self.words[i]

        # text = "갂갃간갠겐관"
        label = [self._char_to_index(c) for c in text]

        return text, label

    def label_to_text(self, label):
        text = ''
        for ll in label:
            text += self._index_to_char(ll)
        return text

    def _random_image(self, height=32, scale=1.0, padding=1):
        text, label = self.random_words()
        image = data_util.synth_image(text, height=height, scale=scale, padding=padding)
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
        label = tf.serialize_sparse(data_util.dense_to_sparse(label, length))
        return image, width, label, length


def _test():
    t = ToySynth()
    print(t.length)
    text, index = t.random_words()
    print(index)
    print("{} : {}".format(text, t.label_to_text(index)))


if __name__ == "__main__":
    _test()
