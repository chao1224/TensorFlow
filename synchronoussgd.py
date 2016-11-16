import tensorflow as tf
import os

worker_num = 5
num_features = 33762578
test_num = 20
iterate_num = 10
break_point = 100
g = tf.Graph()

input_producers = [
    ["./data/tfrecords00", "./data/tfrecords01", "./data/tfrecords02", "./data/tfrecords03", "./data/tfrecords04"],
    ["./data/tfrecords05", "./data/tfrecords06", "./data/tfrecords07", "./data/tfrecords08", "./data/tfrecords09"],
    ["./data/tfrecords10", "./data/tfrecords11", "./data/tfrecords12", "./data/tfrecords13", "./data/tfrecords14"],
    ["./data/tfrecords15", "./data/tfrecords16", "./data/tfrecords17", "./data/tfrecords18", "./data/tfrecords19"],
    ["./data/tfrecords20", "./data/tfrecords21"],
    ["./data/tfrecords22"]
]

with g.as_default():
    with tf.device("/job:worker/task:0"):
        w = tf.Variable(tf.ones([num_features, 1]), name="model")

    for i in range(worker_num):
        with tf.device("/job:worker/task:%d" % i):
            gradients = []
            filename_queue = tf.train.string_input_producer(input_producers[i], num_epochs=None)
            reader = tf.TFRecordReader()
            _, serialized_example = reader.read(filename_queue)
            features = tf.parse_single_example(serialized_example,
                                               features={'label': tf.FixedLenFeature([1], dtype=tf.int64),
                                                         'index': tf.VarLenFeature(dtype=tf.int64),
                                                         'value': tf.VarLenFeature(dtype=tf.float32)})
            label = features['label']
            index = features['index']
            value = features['value']
            dense_feature = tf.sparse_to_dense(tf.sparse_tensor_to_dense(index),
                                               [num_features],
                                               tf.sparse_tensor_to_dense(value))
            y = tf.cast(label, tf.float32)[0]
            x = tf.reshape(dense_feature, shape=[num_features, 1])
            a = tf.transpose(w)
            b = tf.matmul(a, x)
            c = tf.mul(y, b)
            d = tf.mul(y, c - 1)
            local_gradient = tf.mul(d, x)
            gradients.append(tf.mul(local_gradient, 0.01))

    with tf.device("/job:worker/task:0"):
        aggregator = tf.add_n(gradients)
        assign_op = w.assign_add(aggregator)
        w = assign_op

    with tf.device("/job:worker/task:0"):
        filename_queue = tf.train.string_input_producer(input_producers[5], num_epochs=None)
        reader = tf.TFRecordReader()
        _, serialized_example = reader.read(filename_queue)
        features = tf.parse_single_example(serialized_example,
                                           features={'label': tf.FixedLenFeature([1], dtype=tf.int64),
                                                     'index': tf.VarLenFeature(dtype=tf.int64),
                                                     'value': tf.VarLenFeature(dtype=tf.float32)})
        label = features['label']
        index = features['index']
        value = features['value']
        dense_feature = tf.sparse_to_dense(tf.sparse_tensor_to_dense(index),
                                           [num_features],
                                           tf.sparse_tensor_to_dense(value))
        test_y = tf.cast(label, tf.float32)[0]
        test_x = tf.reshape(dense_feature, shape=[num_features, 1])
        test_a = tf.mul(test_y, tf.matmul(tf.transpose(w), test_x))
        print 'test_a:  ', test_a
        test_b = tf.sigmoid(test_a)
        print 'test_b:  ', test_b
        test_c = tf.log(tf.to_float(test_b))
        print 'test_c:  ', test_c
        loss = tf.scalar_mul(-1, test_c)

    with tf.Session("grpc://vm-22-1:2222") as sess:
        sess.run(tf.initialize_all_variables())
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        for i in range(iterate_num):
            output = sess.run(w)
            print output
            if i % break_point == 0 :
                current_loss = 0
                out = open('loss.csv', 'w')
                for j in range(test_num):
                    output2 = sess.run([loss, test_a, test_b, test_c])
                    print "test a: ", output2[1][0],
                    print 'test b: ', output2[2][0],
                    print 'test c: ', output2[3][0],
                    print 'loss: ', output2[0]
                    current_loss += output2[0]
                print >> out, current_loss
                out.close()
        sess.close()

