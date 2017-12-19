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

import model
from config import Config
from data_util import get_bucketed_batch, stv_to_na
from toy_synth import ToySynth
from gamja_synth import GamjaSynth

tf.logging.set_verbosity(tf.logging.DEBUG)


# Non-configurable parameters
# mode = learn.ModeKeys.TRAIN  # 'Configure' training mode for dropout layers

def _get_training(rnn_logits, label, sequence_length):
    with tf.name_scope("train"):
        if Config.tune_scope:
            scope = Config.tune_scope
        else:
            scope = "convnet|rnn"

        rnn_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope)

        loss = model.ctc_loss_layer(rnn_logits, label, sequence_length)

        # Update batch norm stats [http://stackoverflow.com/questions/43234667]
        extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

        with tf.control_dependencies(extra_update_ops):
            learning_rate = tf.train.exponential_decay(Config.learning_rate,
                                                       tf.train.get_global_step(),
                                                       Config.decay_steps,
                                                       Config.decay_rate,
                                                       staircase=Config.decay_staircase,
                                                       name='learning_rate')

            optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate,
                                               beta1=Config.momentum)

            train_op = tf.contrib.layers.optimize_loss(loss=loss,
                                                       global_step=tf.train.get_global_step(),
                                                       learning_rate=learning_rate,
                                                       optimizer=optimizer,
                                                       variables=rnn_vars)

            tf.summary.scalar('learning_rate', learning_rate)

    return train_op


def _get_init_pretrained():
    """Return lambda for reading pretrained initial model"""

    if not Config.tune_from:
        return None

    saver_reader = tf.train.Saver(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES))

    ckpt_path = Config.tune_from

    init_fn = lambda sess: saver_reader.restore(sess, ckpt_path)

    return init_fn


def main(_):
    os.environ['CUDA_VISIBLE_DEVICES'] = "1"
    with tf.Graph().as_default():
        global_step = tf.train.get_or_create_global_step()
        synth = ToySynth()
        with tf.device("/cpu:0"):
            b_image, b_width, b_label, b_length = get_bucketed_batch(synth)

        with tf.device("/gpu:0"):
            features, sequence_length = model.convnet_layers(b_image,
                                                             b_width,
                                                             learn.ModeKeys.TRAIN)
            logits = model.rnn_layers(features,
                                      sequence_length,
                                      synth.num_classes())

            train_op = _get_training(logits, b_label, sequence_length)

        summary_op = tf.summary.merge_all()
        init_op = tf.group(tf.global_variables_initializer(),
                           tf.local_variables_initializer())

        session_config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
        sv = tf.train.Supervisor(logdir=Config.output,
                                 init_op=init_op,
                                 summary_op=summary_op,
                                 save_summaries_secs=Config.save_summaries_secs,
                                 init_fn=_get_init_pretrained(),
                                 save_model_secs=Config.save_model_secs)

        with sv.managed_session(config=session_config) as sess:
            s = time.time()
            while True:
                step, loss = sess.run([global_step, train_op])
                if step % 100 == 0:
                    e = time.time()
                    print("global_step: {}, loss = {}, {}s elapsed".format(step, loss, e - s))
                    s = e


if __name__ == '__main__':
    tf.app.run()
