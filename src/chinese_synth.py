import numpy as np
import random
import tensorflow as tf
import data_util
import simplified_chinese as lang


class ChineseSynth(object):
    def __init__(self, max_char_len=20, max_words=3):
        # for zero padding
        self._num_class = len(lang.alphabet)
        self._num_words = len(lang.words)
        self._max_char_len = max_char_len
        self._max_words = max_words
        self._scales = [1.0, 0.9, 0.8]
        self._width_offset = 5
        self._heigth_offset = 5

    def num_classes(self):
        return self._num_class

    def random_words(self):
        text = ''
        for i in range(random.randint(1, self._max_words)):
            word = random.choice(lang.words)
            if len(text) + len(word) + 1 > self._max_char_len:
                break
            if i > 0:
                text += ' '
            text += word
        index = [lang.rindex[c] for c in text]  # zero padding
        return text, index

    def label_to_text(self, label):
        return "".join([lang.alphabet[l] for l in label])

    def _random_image(self, height=32):
        text, label = self.random_words()
        image = data_util.synth_image(text,
                                      height,
                                      lang.fonts,
                                      self._scales,
                                      self._width_offset,
                                      self._heigth_offset)
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
    t = ChineseSynth()
    _, width, label, length = t._random_image()
    print(width, label, length)
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
