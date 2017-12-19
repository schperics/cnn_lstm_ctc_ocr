import numpy as np
import random
import tensorflow as tf
import data_util


class GamjaSynth(object):
    def __init__(self):
        words, alphabets = data_util.load_word()
        self.words = words
        self.alphabets = alphabets
        self.reverse_index = {}

        index = 0
        for c in alphabets:
            self.reverse_index[c] = index
            index += 1

    def num_classes(self):
        return len(self.alphabets)

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
            text += self.alphabets[i]
        return text

    def _random_image(self, height=32, scale=1.0, padding=1, max_words=3):
        text, label = self.random_words(max_words)
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
    t = GamjaSynth()
    _ = t._random_image()

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
