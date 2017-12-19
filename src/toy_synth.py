from PIL import Image
from PIL import ImageFont
from PIL import ImageDraw
import numpy as np
import random
import re
import tensorflow as tf
import Hangulpy as han
import data_util

k = 5


class ToySynth(object):
    def num_classes(self):
        return k * k * k

    def random_words(self):
        ret = []
        for _ in range(3):
            while True:
                v = random.randint(0, self.num_classes() - 1)
                if (v % 31) == 0:
                    continue
                break
            ret.append(v)
        return ret

    def label_to_text(self, label):
        text = ''
        for ll in label:
            f = ll // (k * k)
            m = (ll % k) // k
            l = ll % k
            text += han.compose(f, m, l)
        return text

    def _random_image(self, height=32, scale=1.0, padding=1):
        label = self.random_words()
        text = self.label_to_text(label)
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
    label = t.random_words()
    print(t.num_classes())
    print(label)
    print(t.label_to_text(label))

    d = t.get()
    with tf.Session() as sess:
        for _ in range(32):
            image, width, label, length = sess.run(d)
            print("image : ", image.shape)
            print("width : ", width)
            print("label : ", label)
            print("length : ", length)


if __name__ == "__main__":
    _test()
