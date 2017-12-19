import tensorflow as tf
from tensorflow.contrib.training import bucket_by_sequence_length
from config import Config
import numpy as np
import random
from PIL import Image
from PIL import ImageFont
from PIL import ImageDraw


def _preprocess_image(image):
    # Rescale from uint8([0,255]) to float([-0.5,0.5])
    image = tf.image.convert_image_dtype(image, tf.float32)
    image = tf.subtract(image, 0.5)

    # Pad with copy of first row to expand to 32 pixels height
    first_row = tf.slice(image, [0, 0, 0], [1, -1, -1])
    image = tf.concat([first_row, image], 0)

    return image


def _get_input_filter(width, length):
    keep_input = None

    if Config.width_threshold != None:
        keep_input = tf.less_equal(width, Config.width_threshold)

    if Config.length_threshold != None:
        length_filter = tf.less_equal(length, Config.length_threshold)
        if keep_input == None:
            keep_input = length_filter
        else:
            keep_input = tf.logical_and(keep_input, length_filter)

    if keep_input == None:
        keep_input = True
    else:
        keep_input = tf.reshape(keep_input, [])  # explicitly make a scalar

    return keep_input


def get_bucketed_batch(synth):
    queue_capacity = Config.num_input_threads * Config.batch_size * 2

    image, width, label, length = synth.get()
    image = _preprocess_image(image)  # move after batch?
    keep_input = _get_input_filter(width, length)

    data_tuple = [image, label, length]
    b_width, b_data_tuple = bucket_by_sequence_length(input_length=width,
                                                      tensors=data_tuple,
                                                      bucket_boundaries=Config.boundaries,
                                                      batch_size=Config.batch_size,
                                                      capacity=queue_capacity,
                                                      keep_input=keep_input,
                                                      allow_smaller_final_batch=True,
                                                      dynamic_pad=True)
    [b_image, b_label, b_length] = b_data_tuple
    b_label = tf.deserialize_many_sparse(b_label, tf.int64)  # post-batching...
    b_label = tf.cast(b_label, tf.int32)  # for ctc_loss

    return b_image, b_width, b_label, b_length


def get_batch(synth, batch_size):
    queue_capacity = Config.num_input_threads * batch_size

    data_tuples = []

    for _ in range(4):
        image, width, label, length = synth.get()
        image = _preprocess_image(image)  # move after batch?
        data_tuples.append([image, width, label, length])

    image, width, label, length = tf.train.batch_join(data_tuples,
                                                      batch_size=batch_size,
                                                      capacity=queue_capacity,
                                                      allow_smaller_final_batch=False,
                                                      dynamic_pad=True)
    label = tf.deserialize_many_sparse(label, tf.int64)  # post-batching...
    label = tf.cast(label, tf.int32)  # for ctc_loss
    return image, width, label, length


def stv_to_na(stv):
    na = np.zeros(stv.dense_shape, np.int32)
    for i, idx in enumerate(stv.indices):
        na[idx[0], idx[1]] = stv.values[i]
    return na


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


def _test():
    import toy_synth, os
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    with tf.device("/cpu:0"):
        t = toy_synth.ToySynth()
        image, width, label, length = get_batch(t)
    with tf.Session() as sess:
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        print(sess.run([label]))
        print(sess.run([label]))
        coord.join(threads)


if __name__ == "__main__":
    _test()
