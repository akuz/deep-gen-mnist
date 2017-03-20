
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from collections import OrderedDict

from tensorflow.examples.tutorials.mnist import input_data

def init(*args, **kwargs):
    return np.random.normal()

if __name__ == "__main__":

    print("Loading data...")
    mnist = input_data.read_data_sets('MNIST_data', one_hot=False)

    batch_size = 10
    image_size = 28

    level_log_z = OrderedDict()
    level_log_z_init = OrderedDict()
    level_z_log_prior = OrderedDict()
    level_z = OrderedDict()
    level_prior = OrderedDict()
    level_entropy = OrderedDict()
    level_w = OrderedDict()
    level_w_init = OrderedDict()

    dimensions = [256, 16, 32, 64, 64]

    with tf.device('/cpu:0'):

        # ----------------------
        # HIDDEN VARIABLE LEVELS
        # ----------------------

        for level in reversed(range(1, len(dimensions))):

            with tf.variable_scope("level_{}".format(level)):

                # level size
                field = 2**(level-1)
                rate = field
                count = image_size - field + 1

                if count < 1:
                    raise ValueError("Hiererachy is too deep, field {}".format(field))

                # level dimensions
                this_dim = dimensions[level]
                lower_dim = dimensions[level-1]
                this_shape = [batch_size, count, count, this_dim]

                with tf.variable_scope("log_z"):

                    rnd_std = 0.1

                    log_z_rnd = tf.random_normal(name='log_z_rnd',
                        mean=0.0, stddev=rnd_std,
                        shape=this_shape)

                    log_z = tf.get_variable(name='log_z', initializer=log_z_rnd)
                    level_log_z[level] = log_z

                    log_z_init = tf.assign(log_z, log_z_rnd, name='log_z_init')
                    level_log_z_init[level] = log_z_init

                    tf.summary.histogram('hist', log_z)

                    log_z_local_prior_dist = tf.contrib.distributions.Normal(
                        0.0, 10.0)

                    log_z_local_prior = tf.reduce_sum(
                        log_z_local_prior_dist.log_prob(log_z))

                    level_prior[level] = -log_z_local_prior


                if level < len(dimensions) - 1:

                    with tf.variable_scope("z_log_prior"):

                        z_log_prior = tf.negative(tf.nn.relu(
                            tf.nn.atrous_conv2d_transpose(
                                level_log_z[level + 1],
                                level_w[level + 1],
                                output_shape=this_shape,
                                rate=rate,
                                padding='VALID')),
                            name='z_log_prior')

                        level_z_log_prior[level] = z_log_prior

                        tf.summary.histogram('hist', z_log_prior)

                    with tf.variable_scope("z"):

                        z = tf.nn.softmax(log_z, name='z')

                        level_z[level] = z

                        tf.summary.histogram('hist', z)

                    with tf.variable_scope("entropy"):

                        entropy = tf.reduce_sum(
                            tf.nn.softmax_cross_entropy_with_logits(
                                labels=z,
                                logits=z_log_prior), 
                            name='entropy')

                        print(entropy)
                        level_entropy[level] = entropy

                with tf.variable_scope("w"):

                    width = 2 if level > 1 else 1

                    w_rnd = tf.random_normal(name='w_rnd', 
                        mean=0.0, stddev=0.01,
                        shape=[width, width, lower_dim, this_dim])

                    w = tf.get_variable(name='w', initializer=w_rnd)
                    level_w[level] = w

                    tf.summary.histogram('hist', w)

                    w_init = tf.assign(w, w_rnd, name='w_init')
                    level_w_init[level] = w_init

        # ----------------------
        # IMAGE COMPARISON LEVEL
        # ----------------------

        level = 0
        with tf.variable_scope("level_{}".format(level)):

            # level dimensions
            this_dim = dimensions[level]
            this_shape = [batch_size, image_size, image_size, this_dim]

            with tf.variable_scope("z_log_prior"):

                z_log_prior = tf.negative(tf.nn.relu(
                    tf.nn.conv2d_transpose(
                        level_log_z[level + 1],
                        level_w[level + 1],
                        strides=[1, 1, 1, 1],
                        output_shape=this_shape,
                        padding='VALID')),
                    name='z_log_prior')

                level_z_log_prior[level] = z_log_prior

                tf.summary.histogram('hist', z_log_prior)

            observed = tf.placeholder(
                shape=[batch_size, image_size * image_size],
                dtype=tf.int32,
                name="observed")

            with tf.variable_scope("entropy"):

                entropy = tf.reduce_sum(
                    tf.nn.sparse_softmax_cross_entropy_with_logits(
                        labels=tf.reshape(observed, [batch_size, image_size, image_size]),
                        logits=z_log_prior),
                    name='entropy')

                level_entropy[level] = entropy

        # -------------
        # TOTAL ENTROPY
        # -------------

        with tf.variable_scope("total"):

            entropies = []

            for level, entropy in level_entropy.items():
                entropies.append(entropy)

            for level, entropy in level_prior.items():
                entropies.append(entropy)

            entropy = tf.add_n(
                entropies,
                name='entropy')

            tf.summary.scalar('entropy', entropy)

        merged_summary = tf.summary.merge_all()
        train_writer = tf.summary.FileWriter('./tmp', tf.get_default_graph())

        # for n in tf.get_default_graph().as_graph_def().node:
        #     print(n.name)

        print("Getting batch...")
        batch = mnist.train.next_batch(batch_size)

        # optimizer also creates variables!
        learning_rate = tf.placeholder(shape=(), dtype=tf.float32, name='learning_rate')
        optimiser = tf.train.AdamOptimizer(learning_rate)
        
        train_step_vars = []
        for _, var in level_log_z.items():
            train_step_vars.append(var)

        train_step = optimiser.minimize(
            entropy, var_list=train_step_vars)

    print("Starting session...")
    with tf.Session() as sess:

        sess.run(tf.global_variables_initializer())

        sess.run(level_log_z_init)
        sess.run(level_w_init)

        start_lr = 0.1
        iter_count = 20
        for i in range(0, iter_count):

            # lr = start_lr - i * (start_lr/iter_count)

            lr = 0.01

            _, entropy_result, summary = sess.run(
                [train_step, entropy, merged_summary], 
                feed_dict={
                    observed: batch[0],
                    learning_rate: lr
                    })

            train_writer.add_summary(summary, i)

            print("{} - entropy: {} - learning rate: {}".format(i+1, entropy_result, lr))

    print("Done")
