import tensorflow as tf
import os

worker_num = 4
g = tf.Graph()
num_features = 33762578

input_producers = [
    ["./data/tfrecords00", "./data/tfrecords01", "./data/tfrecords02", "./data/tfrecords03", "./data/tfrecords04"]
]

with g.as_default():
    with tf.device("/job:worker/task:0"):
        w = tf.Variable(tf.ones([num_features, 1]), name="model")

    with tf.device("/job:worker/task:0"):
        gradients = []
        filename_queue = tf.train.string_input_producer(input_producers[0], num_epochs=None)
        reader = tf.TFRecordReader()
        _, serialized_example = reader.read(filename_queue)
        features = tf.parse_single_example(serialized_example,
                                           features={
                                            'label': tf.FixedLenFeature([1], dtype=tf.int64),
                                            'index': tf.VarLenFeature(dtype=tf.int64),
                                            'value': tf.VarLenFeature(dtype=tf.float32),
                                           }
                                          )
        label = features['label']
        index = features['index']
        value = features['value']
        dense_feature = tf.sparse_to_dense(tf.sparse_tensor_to_dense(index),
                                           [num_features],
                                           tf.sparse_tensor_to_dense(value))
        y = tf.cast(label, tf.float32)[0]
        print('y:  ', y)
        x = tf.reshape(dense_feature, shape=[num_features, 1])
        print('x: ', x)
        a = tf.transpose(w)
        print('a: ', a)
        b = tf.matmul(a, x)
        print('b: ', b)
        c = tf.mul(y, b)
        d = tf.mul(y, c - 1)
        local_gradient = tf.mul(d, x)
        gradients.append(tf.mul(local_gradient, 0.01))

    # we create an operator to aggregate the local gradients
    with tf.device("/job:worker/task:0"):
        aggregator = tf.add_n(gradients)
        assign_op = w.assign_add(aggregator)

    with tf.Session("grpc://vm-22-1:2222") as sess:
        # sess.run(tf.initialize_all_variables())
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        for i in range(10):
            output = sess.run([x, y, assign_op])
            print output[0].shape,
            print output[1].shape,
            print output[2].shape