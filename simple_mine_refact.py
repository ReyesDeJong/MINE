from __future__ import division
from __future__ import print_function

import tensorflow as tf

from grad_corrected_mine import GradMINE


class SimpleMINE(GradMINE):

    def __init__(self, params, name = "simple_mine"):
        super().__init__(params, name)


    
    def _optimizer_init(self, loss, lr):
        with tf.name_scope("optimizer"):
            stat_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="stat_net")
            optimizer = tf.train.AdamOptimizer(learning_rate=lr)
            train_step = optimizer.minimize(-loss, var_list=stat_vars)  # Maximization <=> Neg minimization
        return train_step