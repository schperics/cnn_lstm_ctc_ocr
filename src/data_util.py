import tensorflow as tf
from tensorflow.contrib.training import bucket_by_sequence_length
from config import Config
import numpy as np


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

def stv_to_na(stv) :
    na = np.zeros(stv.dense_shape, np.int32)
    for i, idx in enumerate(stv.indices) :
        na[idx[0], idx[1]] = stv.values[i]
    return na


def _test():
    import toy_synth
    with tf.device("/cpu:0"):
        t = toy_synth.ToySynth()
        image, width, label, length = get_bucketed_batch(t)
    with tf.Session() as sess:
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        print(sess.run([length]))
        print(sess.run([length]))
        print(sess.run([length]))
        print(sess.run([length]))
        print(sess.run([length]))
        coord.join(threads)


if __name__ == "__main__":
    _test()
