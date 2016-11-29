import tensorflow as tf
import os

worker_num = 5
num_features = 33762578
iterate_num = 3
test_num = 50
break_point = 3
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
        w = tf.Variable(tf.random_normal([num_features, 1]), name="model")
        # mmm = tf.ones([num_features, 1])

    gradients = []
    y_list = []
    for i in range(1):
        with tf.device("/job:worker/task:%d" % i):
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
            c = tf.sigmoid(tf.mul(y, b))
            d = tf.mul(y, c - 1)
            local_gradient = tf.mul(d, x)
            y_list.append(y)
            gradients.append(tf.mul(local_gradient, 0.05))

    with tf.device("/job:worker/task:0"):
        aggregator = tf.add_n(gradients)
        assign_op = w.assign_add(aggregator)

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
        test_y = tf.cast(label, tf.float32)
        test_x = tf.reshape(dense_feature, shape=[num_features, 1])

        # test_a = tf.scalar_mul(test_y, tf.matmul(tf.transpose(w), test_x))
        # test_b = tf.sigmoid(test_a)
        # test_c = tf.log(test_b)
        # loss = tf.scalar_mul(-1, test_c)
        # # norm = tf.matmul(tf.transpose(w), mmm)
        # test_d = tf.exp(tf.scalar_mul(-1, test_a))
        # one = tf.convert_to_tensor([[1.0]])
        # test_e = tf.add(one, test_d)
        # loss2 = tf.log(test_e)

        predict_confidence = tf.matmul(tf.transpose(w), test_x)
        predict_y = tf.sign(predict_confidence)[0]
        cnt = tf.equal(test_y, predict_y)
        norm = tf.matmul(tf.transpose(w), w)


    with tf.Session("grpc://vm-22-1:2222") as sess:
        sess.run(tf.initialize_all_variables())
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        for i in range(1, 1+iterate_num):
            print 'iteration {}/{}'.format(i, iterate_num)
            output = sess.run([assign_op])
            if i % break_point == 0:
                current_error = 0
                out = open('error_syn.csv', 'a')
                for j in range(test_num):
                    output2 = sess.run([test_y, predict_y, cnt, norm])
                    is_right = output2[2][0]
                    if not is_right:
                        current_error += 1
                    # print 'test_y:  ', output2[0],
                    # print '\tpredict_y:  ', output2[1],
                    # print '\terror:  ', current_error,
                    # print '\tnorm:  ', output2[3]
                print 'error: ', current_error
                #print >> out, current_error
                out.close()
        sess.close()