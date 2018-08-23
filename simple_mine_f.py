from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import time
import datetime

from grad_corrected_mine import GradMINE


class MINEf(GradMINE):
    def __init__(self, params, name="mine_f"):
        tf.reset_default_graph()

        self.name = name
        self.p = self._fix_to_default(params)

        # Directories
        self.model_path, self.ckpt_path, self.tb_path = self._create_directories()

        # Input pipeline
        self.iterator, self.inputs_x_ph, self.inputs_z_ph = self._iterator_init(self.p["batch_size"])

        self.loss, self.train_step = self._build_graph(self.iterator, self.p["learning_rate"])

        # Init
        config = tf.ConfigProto(allow_soft_placement=True)
        config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=config)
        self.sess.run(tf.global_variables_initializer())

    def train(self, inputs_x, inputs_z, max_it, stat_every):
        # Init iterator with trainning data
        self.sess.run(self.iterator.initializer, feed_dict={self.inputs_x_ph: inputs_x, self.inputs_z_ph: inputs_z})
        # Initialization of log and saver
        saver = tf.train.Saver(keep_checkpoint_every_n_hours=1)
        summ_writer = tf.summary.FileWriter(self.tb_path, self.sess.graph)
        merged = tf.summary.merge_all()
        start_time = time.time()
        print("Beginning training of " + self.name)
        for it in range(1, max_it + 1):
            _, train_loss, summ = self.sess.run([self.train_step, self.loss, merged])
            if it % stat_every == 0:
                elapsed_time = time.time() - start_time
                print("Iteration %i / %i: loss %f -- elapsed time %f [s]" % (it, max_it, train_loss, elapsed_time),
                      flush=True)
                summ_writer.add_summary(summ, it)
        save_path = saver.save(self.sess, self.ckpt_path, global_step=max_it)
        print("Model saved to: %s" % save_path)

    def _fix_to_default(self, params):
        if "batch_size" not in params:
            params["batch_size"] = 256
        if "learning_rate" not in params:
            params["input_dim"] = 1e-4
        if "input_dim" not in params:
            raise AttributeError("Dimensions of input needed")
        if "ema_decay" not in params:
            params["ema_decay"] = 0.999
        return params

    def _create_directories(self):
        date = datetime.datetime.now().strftime("%Y%m%d")
        self.model_path = "results/" + self.name + "_" + date + "/"
        self.ckpt_path = self.model_path + "ckpt/model"
        self.tb_path = self.model_path + "tb_summ/"
        # Delete previus content of tensorboard logs
        if tf.gfile.Exists(self.tb_path):
            tf.gfile.DeleteRecursively(self.tb_path)
        return self.model_path, self.ckpt_path, self.tb_path

    def _iterator_init(self, batch_size):
        with tf.device('/cpu:0'):
            with tf.name_scope("input"):
                inputs_x_ph = tf.placeholder(dtype=tf.float32, shape=[None, self.p["input_dim"]])
                inputs_z_ph = tf.placeholder(dtype=tf.float32, shape=[None, self.p["input_dim"]])
                # Dataset
                dataset = tf.data.Dataset.from_tensor_slices((inputs_x_ph, inputs_z_ph))
                dataset = dataset.repeat()
                dataset = dataset.shuffle(buffer_size=5000)
                dataset = dataset.batch(batch_size=batch_size)
                dataset = dataset.prefetch(buffer_size=4)
                # Iterator
                iterator = dataset.make_initializable_iterator()
        return iterator, inputs_x_ph, inputs_z_ph

    def _build_graph(self, iterator, lr):
        x_it, z_it = iterator.get_next()
        _, z_hat_it = iterator.get_next()
        # Inputs
        self.x = tf.placeholder_with_default(x_it, shape=[None, self.p["input_dim"]], name="x")
        self.z = tf.placeholder_with_default(z_it, shape=[None, self.p["input_dim"]], name="z")
        self.z_hat = tf.placeholder_with_default(z_hat_it, shape=[None, self.p["input_dim"]], name="z_hat")

        # Model
        with tf.name_scope("stat_net_t"):
            out_t = self._stat_net(self.x, self.z)
        with tf.name_scope("stat_net_t_prime"):
            out_t_prime = self._stat_net(self.x, self.z_hat, reuse=True)
        tf.summary.histogram("out_t", out_t)
        tf.summary.histogram("out_t_prime", out_t_prime)

        loss, self.term_1, self.term_2 = self._loss_init(out_t, out_t_prime)

        train_step = self._optimizer_init(loss, lr)
        return loss, train_step


    def _loss_init(self, out_t, out_t_prime):
        with tf.name_scope("loss"):
            term_1 = tf.reduce_mean(out_t)
            term_2 = tf.reduce_mean(tf.exp(out_t_prime - 1))
            loss = term_1 - term_2

            tf.summary.scalar("term_1", term_1)
            tf.summary.scalar("term_2", term_2)
            tf.summary.scalar("mine_loss", loss)
        return loss, term_1, term_2

    def _optimizer_init(self, loss, lr):
        with tf.name_scope("optimizer"):
            self.stat_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="stat_net")
            self.optimizer = tf.train.AdamOptimizer(learning_rate=lr, epsilon=1e-6)
            train_step = self.optimizer.minimize(-loss, var_list=self.stat_vars)  # Maximization <=> Neg minimization

            # corrected_gradients = self.gradient_bias_correction()
            # train_step = self.optimizer.apply_gradients(zip(corrected_gradients, self.stat_vars)) #train_step = optimizer.minimize(-loss, var_list=stat_vars)  # Maximization <=> Neg minimization
            # train_step = self.optimizer.minimize(-loss, var_list=self.stat_vars)  # Maximization <=> Neg minimization
        return train_step