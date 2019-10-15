import sys
import os
import pytest
import numpy as np
import time

import horovod.tensorflow as hvd
import tensorflow as tf


def broadcast(device, sess, dtype=tf.float32, shape=[5000, 10000], it=1):
    data_res = tf.random_uniform(shape, seed=0, dtype=dtype)
    # gpu broadcast
    with tf.device("/{}:0".format(device)):
        data_tf = tf.random_uniform(shape, seed=hvd.rank(), dtype=dtype, name='data_tf')

    r_expected = sess.run(data_res)
    for i in range(it):
        with tf.device("/{}:0".format(device)):
            res = hvd.broadcast(data_tf, 0)

        r = sess.run(res)
        if i == 0: assert(np.allclose(r, r_expected))


def allreduce(device, sess, dtype=tf.float32, shape=[5000, 10000], it=1):
    data_res = tf.zeros(shape, dtype=dtype)
    for i in range(hvd.size()):
        data_res += tf.random_uniform(shape, seed=i, dtype=dtype)
    # gpu broadcast
    with tf.device("/{}:0".format(device)):
        data_tf = tf.random_uniform(shape, seed=hvd.rank(), dtype=dtype, name='data_tf')

    r_expected = sess.run(data_res)

    res = data_tf
    for i in range(it):
        with tf.device("/{}:0".format(device)):
            a = hvd.allreduce(res, average=False)
            r = sess.run(a)

    # r = sess.run(res)
    # if i == 0: assert(np.allclose(r, r_expected))


def allgather(device, sess, dtype=tf.float32, shape=[5000, 10000], it=1):
    data_res = tf.zeros(shape, dtype=dtype)
    for i in range(hvd.size()):
        data_res += tf.random_uniform(shape, seed=i, dtype=dtype)
    data_res = tf.math.reduce_sum(data_res)

    # gpu broadcast
    with tf.device("/{}:0".format(device)):
        data_tf = tf.random_uniform(shape, seed=hvd.rank(), dtype=dtype, name='data_tf')

    r_expected = sess.run(data_res)

    for i in range(it):
        with tf.device("/{}:0".format(device)):
            res = hvd.allgather(data_tf)
            res = tf.math.reduce_sum(res)

        r = sess.run(res)
        # print(r, r_expected)
        if i == 0: assert(r == r_expected or int(r) == int(r_expected) or int(r / 10000) == int(r_expected / 10000))


def test_operations():
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.gpu_options.visible_device_list = str(hvd.local_rank())
    sess = tf.Session(config=config)

    devices = ['gpu', 'cpu']
    dtypes = [tf.float32, tf.float64, tf.float16]
    shapes = [[500, 10000], [1000, 10000]]
    for shape in shapes:
        for device in devices:
            for dtype in dtypes:
                broadcast(device, sess, dtype=dtype, shape=shape)
                allreduce(device, sess, dtype=dtype, shape=shape)
                if device == 'cpu':
                    allgather(device, sess, dtype=dtype, shape=shape)

def test_perf():
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.gpu_options.visible_device_list = str(hvd.local_rank())
    sess = tf.Session(config=config)

    devices = ['gpu', 'cpu']
    dtype = tf.float32
    shapes = [[500000, 250]]  # [[1, 250], [1000, 250], [500000, 250]]
    ops = ['allreduce']  # ['broadcast', 'allreduce', 'allgather']
    it = 100
    if hvd.rank() == 0:
        stream = open("res.csv", "a")
    for op in ops:
        for device in devices:
            print("Benchmarking: {} {}".format(device, op))
            for shape in shapes:
                if (op == 'allgather' or op == 'broadcast') and device == 'gpu':
                    continue

                start = time.time()
                if op == 'broadcast':
                    broadcast(device, sess, dtype, shape, it=it)
                elif op == 'allreduce':
                    allreduce(device, sess, dtype, shape, it=it)
                else:
                    allgather(device, sess, dtype, shape, it=it)
                end = time.time()
                if hvd.rank() == 0:
                    stream.write('{},{},{},{},{}\n'.format(op, hvd.size(), shape[0] * shape[1] * 4, device, (end - start) * 1000 / float(it)))

    if hvd.rank() == 0:
        stream.close()


if __name__ == "__main__":
    hvd.init()
    test_perf()
