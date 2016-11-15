"""
A solution to finding trace of square of a large matrix using a single device.
We are able to circumvent OOM errors, by generating sub-matrices. TensorFlow
runtime, is able to schedule computation on small sub-matrices without
overflowing the available RAM.
"""

import tensorflow as tf
import os

server_num = 4
N = 100000 # dimension of the matrix
d = 10 # number of splits along one dimension. Thus, we will have 100 blocks
M = int(N / d)

def get_block_name(i, j):
    return "sub-matrix-"+str(i)+"-"+str(j)


def get_intermediate_trace_name(i, j):
    return "inter-"+str(i)+"-"+str(j)

def get_row_name(i):
    return "row-"+str(i)

def block_method():
    g = tf.Graph()

    with g.as_default():
        matrices = {}
        for i in range(0, d):
            for j in range(0, d):
                with tf.device("/job:worker/task:%d" % ((i*(d-1)+j) % server_num)):
                    matrix_name = get_block_name(i, j)
                    matrices[matrix_name] = tf.random_uniform([M, M], name=matrix_name)

        intermediate_traces = {}
        for i in range(0, d):
            for j in range(0, d):
                with tf.device("/job:worker/task:%d" % ((i*(d-1)+j) % server_num)):
                    A = matrices[get_block_name(i, j)]
                    B = matrices[get_block_name(j, i)]
                    intermediate_traces[get_intermediate_trace_name(i, j)] = tf.trace(tf.matmul(A, B))

        with tf.device("/job:worker/task:0"):
            retval = tf.add_n(intermediate_traces.values())

        config = tf.ConfigProto(log_device_placement=True)
        with tf.Session("grpc://vm-22-2:2222", config=config) as sess:
            result = sess.run(retval)
            sess.close()
            print result

def speedUp():
    g = tf.Graph()

    row_number = N

    with g.as_default():
        matrices = {}
        for r in range(row_number):
            with tf.device("/job:worker/task:%d" % (r % server_num)):
                row_name = get_row_name(r)
                matrices[row_name] = tf.random_uniform([1, N], name=row_name)

        for r in range(row_number):
            with tf.device("/job:worker/task:%d" % (r % server_num)):
                A = matrices[get_row_name(r)]
                intermediate_traces[get_row_name(r)] = tf.matmul(A, tf.transpose(A))

        with tf.device("/job:worker/task:0"):
            retval = tf.add_n(intermediate_traces.values())

        config = tf.ConfigProto(log_device_placement=True)
        with tf.Session("grpc://vm-22-2:2222", config=config) as sess:
            result = sess.run(retval)
            sess.close()
            print result

speedUp()