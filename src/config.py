import tensorflow as tf

Config = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('output', '../data/model',
                           """Directory for event logs and checkpoints""")
tf.app.flags.DEFINE_string('tune_from', '',
                           """Path to pre-trained model checkpoint""")
tf.app.flags.DEFINE_string('tune_scope', '',
                           """Variable scope for training""")

tf.app.flags.DEFINE_integer('batch_size', 2 ** 5,
                            """Mini-batch size""")
tf.app.flags.DEFINE_float('learning_rate', 1e-4,
                          """Initial learning rate""")
tf.app.flags.DEFINE_float('momentum', 0.9,
                          """Optimizer gradient first-order momentum""")
tf.app.flags.DEFINE_float('decay_rate', 0.9,
                          """Learning rate decay base""")
tf.app.flags.DEFINE_float('decay_steps', 2 ** 16,
                          """Learning rate decay exponent scale""")
tf.app.flags.DEFINE_float('decay_staircase', False,
                          """Staircase learning rate decay by integer division""")

tf.app.flags.DEFINE_integer('max_num_steps', 2 ** 21,
                            """Number of optimization steps to run""")

tf.app.flags.DEFINE_string('train_device', '/gpu:1',
                           """Device for training graph placement""")
tf.app.flags.DEFINE_string('input_device', '/gpu:0',
                           """Device for preprocess/batching graph placement""")

tf.app.flags.DEFINE_string('train_path', '../data/train/',
                           """Base directory for training data""")
tf.app.flags.DEFINE_string('filename_pattern', 'words-*',
                           """File pattern for input data""")
tf.app.flags.DEFINE_integer('num_input_threads', 4,
                            """Number of readers for input data""")
tf.app.flags.DEFINE_integer('width_threshold', None,
                            """Limit of input image width""")
tf.app.flags.DEFINE_integer('length_threshold', None,
                            """Limit of input string length width""")
tf.app.flags.DEFINE_integer('save_summaries_secs', 600, "")
tf.app.flags.DEFINE_integer('save_model_secs', 1800, "")


Config.boundaries=[32, 64, 96, 128, 160, 192, 224, 256]

if __name__ == "__main__" :
    print(Config.boundaries)
    print(Config.batch_size)