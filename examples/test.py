import os
import errno
import tensorflow as tf
import horovod.tensorflow as hvd
import numpy as np
import time

from tensorflow import keras

layers = tf.layers

tf.logging.set_verbosity(tf.logging.INFO)

rank = hvd.rank()

hvd.init(dummy=False)

config = tf.ConfigProto()
# config.allow_soft_placement = True
config.gpu_options.allow_growth = True
config.gpu_options.visible_device_list = str(rank)
# config.log_device_placement=True

sess = tf.Session(config=config)

with tf.device("/gpu:0"):
    if rank == 0:
        # a = tf.get_variable('test_tensor', [500, 1000], initializer=tf.initializers.random_normal)
        a = tf.random_uniform([500, 1000], seed=1, dtype=tf.float32, name='a')
    else:
        # a = tf.get_variable('test_tensor', [500, 1000], initializer=tf.initializers.random_normal)
        a = tf.random_uniform([500, 1000], seed=2, dtype=tf.float32, name='a')

sess.run(tf.global_variables_initializer())

y = hvd.allreduce(a)
# y = hvd.allgather(y)
y = tf.math.reduce_sum(y)
res = sess.run(y)

print("[{}] a = {}".format(rank, res))

writer = tf.summary.FileWriter('./graphs', sess.graph)
