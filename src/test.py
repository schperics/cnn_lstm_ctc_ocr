# CNN-LSTM-CTC-OCR
# Copyright (C) 2017 Jerod Weinman
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

import os
import time
import tensorflow as tf
from tensorflow.contrib import learn
from time import gmtime, strftime
import model
import data_util
from toy_synth import ToySynth
from gamja_synth import GamjaSynth
from config import Config

tf.logging.set_verbosity(tf.logging.WARN)

# Non-configurable parameters
mode = learn.ModeKeys.INFER  # 'Configure' training mode for dropout layers


def _get_session_config():
    """Setup session config to soften device placement"""
    config = tf.ConfigProto(
        allow_soft_placement=True,
        log_device_placement=False)

    return config


def _get_testing(rnn_logits, sequence_length, label, label_length):
    """Create ops for testing (all scalars): 
       loss: CTC loss function value, 
       label_error:  Batch-normalized edit distance on beam search max
       sequence_error: Batch-normalized sequence error rate
    """
    with tf.name_scope("train"):
        loss = model.ctc_loss_layer(rnn_logits, label, sequence_length)
    with tf.name_scope("test"):
        predictions, _ = tf.nn.ctc_beam_search_decoder(rnn_logits,
                                                       sequence_length,
                                                       beam_width=1,
                                                       top_paths=1,
                                                       merge_repeated=True)
        hypothesis = tf.cast(predictions[0], tf.int32)  # for edit_distance
        label_errors = tf.edit_distance(hypothesis, label, normalize=False)
        sequence_errors = tf.count_nonzero(label_errors, axis=0)
        total_label_error = tf.reduce_sum(label_errors)
        total_labels = tf.reduce_sum(label_length)
        label_error = tf.truediv(total_label_error,
                                 tf.cast(total_labels, tf.float32),
                                 name='label_error')
        sequence_error = tf.truediv(tf.cast(sequence_errors, tf.int32),
                                    tf.shape(label_length)[0],
                                    name='sequence_error')
        # tf.summary.scalar('loss', loss)
        # tf.summary.scalar('label_error', label_error)
        # tf.summary.scalar('sequence_error', sequence_error)

    return loss, label_error, sequence_error, predictions[0]


def _get_checkpoint():
    ckpt = tf.train.get_checkpoint_state(Config.output)

    if ckpt and ckpt.model_checkpoint_path:
        ckpt_path = ckpt.model_checkpoint_path
    else:
        raise RuntimeError('No checkpoint file found')

    return ckpt_path


def _get_init_trained():
    """Return init function to restore trained model from a given checkpoint"""
    saver_reader = tf.train.Saver(
        tf.get_collection(tf.GraphKeys.GLOBAL_STEP) +
        tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
    )

    init_fn = lambda sess, ckpt_path: saver_reader.restore(sess, ckpt_path)
    return init_fn


def main(_):
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    synth = ToySynth()
    with tf.Graph().as_default():
        with tf.device("/cpu:0"):
            image, width, label, length = data_util.get_batch(synth, 2 ** 8)

        with tf.device("/gpu:0"):
            features, sequence_length = model.convnet_layers(image, width, mode)
            logits = model.rnn_layers(features, sequence_length, synth.num_classes())
            loss, label_error, sequence_error, pred = _get_testing(logits, sequence_length, label, length)

        global_step = tf.contrib.framework.get_or_create_global_step()

        session_config = _get_session_config()
        restore_model = _get_init_trained()

        # summary_op = tf.summary.merge_all()
        init_op = tf.group(tf.global_variables_initializer(),
                           tf.local_variables_initializer())

        # summary_writer = tf.summary.FileWriter(os.path.join(Config.output, "test"))

        step_ops = [global_step, loss, label_error, sequence_error, label, pred]

        with tf.Session(config=session_config) as sess:
            sess.run(init_op)
            coord = tf.train.Coordinator()  # Launch reader threads
            threads = tf.train.start_queue_runners(sess=sess, coord=coord)
            # summary_writer.add_graph(sess.graph)

            while True:
                print("resotre models")
                restore_model(sess, _get_checkpoint())  # Get latest checkpoint
                s = time.time()
                print("calculating...")
                step, loss_val, ce, we, l, p = sess.run(step_ops)
                print("elapsed {}s".format(time.time() - s))
                print("global_step={}, loss={}, ce={}, we={}".format(
                    step, loss_val, ce, we
                ))
                p = data_util.stv_to_na(p)
                l = data_util.stv_to_na(l)
                for i in range(len(l)):
                    print("{} : {}".format(synth.label_to_text(l[i]),
                                           synth.label_to_text(p[i])))

                """
                summary_str = sess.run(summary_op)
                summary_writer.add_summary(summary_str, step)
                """

                print("{}. wait for {}sec".format(
                    strftime("%Y-%m-%d %H:%M:%S", gmtime()),
                    Config.save_model_secs + 10
                ))
                time.sleep(Config.save_model_secs + 10)

            coord.join(threads)


if __name__ == '__main__':
    tf.app.run()
