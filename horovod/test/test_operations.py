import sys
import os
import pytest
import numpy as np

import horovod.tensorflow as hvd
import tensorflow as tf


def broadcast(device, dtype=tf.float32, shape=[5000, 10000]):
    data_res = tf.random_uniform(shape, seed=0, dtype=dtype)
    # gpu broadcast
    with tf.device("/{}:0".format(device)):
        data_tf = tf.random_uniform(shape, seed=hvd.rank(), dtype=dtype, name='data_tf')
        res = hvd.broadcast(data_tf, 0)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.gpu_options.visible_device_list = str(hvd.local_rank())
    # config.log_device_placement=True

    with tf.Session(config=config) as sess:
        r = sess.run(res)
        r_expected = sess.run(data_res)
        assert(np.array_equal(r, r_expected))


def allreduce(device, dtype=tf.float32, shape=[5000, 10000]):
    data_res = tf.zeros(shape, dtype=dtype)
    for i in range(hvd.size()):
        data_res += tf.random_uniform(shape, seed=i, dtype=dtype)
    data_res /= hvd.size()
    # gpu broadcast
    with tf.device("/{}:0".format(device)):
        data_tf = tf.random_uniform(shape, seed=hvd.rank(), dtype=dtype, name='data_tf')
        res = hvd.allreduce(data_tf)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.gpu_options.visible_device_list = str(hvd.local_rank())
    # config.log_device_placement=True

    with tf.Session(config=config) as sess:
        r = sess.run(res)
        r_expected = sess.run(data_res)
        assert(np.array_equal(r, r_expected))


def allgather(device, dtype=tf.float32, shape=[5000, 10000]):
    data_res = tf.zeros(shape, dtype=dtype)
    for i in range(hvd.size()):
        data_res += tf.random_uniform(shape, seed=i, dtype=dtype)
    data_res = tf.math.reduce_sum(data_res)

    # gpu broadcast
    with tf.device("/{}:0".format(device)):
        data_tf = tf.random_uniform(shape, seed=hvd.rank(), dtype=dtype, name='data_tf')
        res = hvd.allgather(data_tf)
        res = tf.math.reduce_sum(res)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.gpu_options.visible_device_list = str(hvd.local_rank())
    # config.log_device_placement=True

    with tf.Session(config=config) as sess:
        r = sess.run(res)
        r_expected = sess.run(data_res)
        print(r, r_expected)
        assert(r == r_expected or int(r) == int(r_expected) or int(r / 1000) == int(r_expected / 1000))


def test_operations():
    devices = ['gpu', 'cpu']
    dtypes = [tf.float32, tf.float64, tf.float16]
    shapes = [[500, 10000], [1000, 10000]]
    for shape in shapes:
        for device in devices:
            for dtype in dtypes:
                broadcast(device, dtype, shape)
                allreduce(device, dtype, shape)
                if device == 'cpu':
                    allgather(device, dtype, shape)
