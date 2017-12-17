import os
import tensorflow as tf
import pickle
import math
import numpy as np
from sklearn.model_selection import train_test_split

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_string('save_model_dir',
                    'tmp/',
                    'path to the directory for saving the model')

flags.DEFINE_integer('batch_size', 256, 'Batch size of the model')
flags.DEFINE_integer('num_epochs', 1000, 'Number of the epochs in training')
flags.DEFINE_float('learning_rate', 1e-4, 'learning rate of the optimizer')
flags.DEFINE_integer('embedding_size', 128,
                     'Embedding size of the icd encoder')
flags.DEFINE_integer('num_sampled', 64,
                     'Number of negative examples to sample.')


def load_data():
    data_table = pickle.load(open('data/word2vec_table.p', 'rb'))
    data_table = np.array(data_table)

    word_id = pickle.load(open('data/word_id.p', 'rb'))
    unique_word = word_id.keys()
    num_words = len(unique_word) + 1

    return data_table, num_words


def build_iter_ops(data_table, vocabulary_size, train=True, reuse=False):
    data_tuple = (data_table[:, 0], data_table[:, 1], data_table[:, 2])
    dataset = tf.data.Dataset.from_tensor_slices(data_tuple)
    if train:
        dataset = dataset.shuffle(buffer_size=FLAGS.batch_size * 6)
    dataset = dataset.batch(FLAGS.batch_size)
    iterator = dataset.make_initializable_iterator()
    x, y, w = iterator.get_next()
    ops = word2vec_net(x, y, w, vocabulary_size, reuse, train)

    return iterator, ops


def word2vec_net(x, y, w, vocabulary_size, reuse=False, train=True):
    with tf.variable_scope('word2vec', reuse=reuse):
        y = tf.reshape(y, [-1, 1])
        w = tf.cast(w, tf.float32)
        embed_matrix = tf.get_variable(
            'embedding_matrix', dtype=tf.float32,
            shape=[vocabulary_size, FLAGS.embedding_size],
            initializer=tf.random_uniform_initializer(-1.0, 1.0),
        )

        softmax_weights = tf.get_variable(
            name='softmax_weight', dtype=tf.float32,
            shape=[vocabulary_size, FLAGS.embedding_size],
            initializer=tf.truncated_normal_initializer(
                stddev=1.0 / math.sqrt(FLAGS.embedding_size))
        )

        softmax_biases = tf.get_variable(
            name='softmax_bias', initializer=tf.zeros_initializer(),
            shape=[vocabulary_size], dtype=tf.float32
        )

    embed = tf.nn.embedding_lookup(embed_matrix, x)

    losses = tf.nn.sampled_softmax_loss(weights=softmax_weights,
                                        biases=softmax_biases, inputs=embed,
                                        labels=y,
                                        num_sampled=FLAGS.num_sampled,
                                        num_classes=vocabulary_size)

    step_loss = tf.losses.compute_weighted_loss(losses, weights=w)
    if train:
        train_step = tf.train.AdamOptimizer(0.001).minimize(step_loss)
        return losses * w, train_step
    else:
        return losses * w,


def run_test(sess, iterator, ops):
    sess.run(tf.local_variables_initializer())
    sess.run(iterator.initializer)
    epoch_losses = []

    while True:
        try:
            op_result = sess.run(ops)
            epoch_losses += list(op_result[0])

        except tf.errors.OutOfRangeError:
            break

    return np.mean(epoch_losses)


def main(_):
    data_table, vocabulary_size = load_data()

    train_data, val_data = train_test_split(
        data_table, test_size=0.20, random_state=1)

    with tf.Graph().as_default():
        train_iter, train_ops = build_iter_ops(
            train_data, vocabulary_size, reuse=False, train=True)
        val_iter, val_ops = build_iter_ops(
            val_data, vocabulary_size, reuse=True, train=False)

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            for iter_num in range(FLAGS.num_epochs):
                train_losses = run_test(sess, train_iter, train_ops)
                val_losses = run_test(sess, val_iter, val_ops)

                print(iter_num, train_losses, val_losses)


if __name__ == '__main__':
    tf.app.run()
