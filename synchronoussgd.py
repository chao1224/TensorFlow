import tensorflow as tf
import os

worker_num = 5
num_features = 33762578
iterate_num = 200000
test_num = 1000
break_point = 100000
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
        w = tf.Variable(tf.random_normal([num_features]), name="model")

    # 1. update indice to parameter server
    value_list = [0, 0, 0, 0, 0]
    index_list = [0, 0, 0, 0, 0]
    label_list = [0, 0, 0, 0, 0]
    for i in range(worker_num):
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
            label_list[i] = label
            index_list[i] = index
            value_list[i] = value

    # 2. push dense weights to all workers
    weight_list = [0, 0, 0, 0, 0]
    with tf.device("/job:worker/task:0"):
        for i in range(worker_num):
            index = index_list[i]
            # if i == 0:
            #     print 'values:  ', index.values
            weight_list[i] = tf.gather(w, index.values)

    # 3. calculate gradients
    gradients = [0, 0, 0, 0, 0]
    ooo = []
    for i in range(worker_num):
        with tf.device("/job:worker/task:%d" % i):
            weight = weight_list[i]
            value = value_list[i]

            y = tf.cast(label_list[i], tf.float32)[0]
            x = value.values
            # a = tf.transpose(weight)
            # b = tf.matmul(a, weight)
            a = tf.mul(weight, x)

            if i == 0:
                print 'weight   ', weight
                print 'x  ', x
                print 'a ', a

            b = tf.reduce_sum(a)
            c = tf.sigmoid(tf.mul(y, b))
            d = tf.mul(y, c-1)
            ooo.append(d)
            local_gradient = tf.mul(d, x)
            gradients[i] = tf.mul(local_gradient, 0.1)

    # 4. update gradients
    with tf.device("/job:worker/task:0"):
        mark = ''
        aaa = gradients
        bbb = index_list
        ccc = ooo

        print 'gradient  ', gradients[0]
        print 'index   ', index_list[0].values

        # len0 = index_list[0].shape
        # len_0 = len0[0]

        # gg = tf.IndexedSlices(gradients[0], index_list[0].values)
        # assign_0 = w.scatter_sub(gg)

        # i0 = tf.reshape(index_list[0].values, shape=[len_0, 1])
        # g0 = tf.reshape(gradients[0], shape=[len_0, 1])
        # assign_0 = tf.scatter_sub(w[:, 0],i0, g0)

        op1 = tf.scatter_sub(w, index_list[0].values, gradients[0])
        op2 = tf.scatter_sub(w, index_list[1].values, gradients[1])
        op3 = tf.scatter_sub(w, index_list[2].values, gradients[2])
        op4 = tf.scatter_sub(w, index_list[3].values, gradients[3])
        op5 = tf.scatter_sub(w, index_list[4].values, gradients[4])

        # aggregator = tf.add_n(gradients)
        # assign_op = w.scatter_sub(aggregator)

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
        # test_x = tf.reshape(dense_feature, shape=[num_features, 1])
        test_x = dense_feature

        predict_confidence = tf.reduce_sum(tf.mul(w, test_x))
        predict_y = tf.sign(predict_confidence)
        cnt = tf.equal(test_y, predict_y)
        norm = tf.reduce_sum(tf.mul(w, w))

    with tf.Session("grpc://vm-22-1:2222") as sess:
        sess.run(tf.initialize_all_variables())
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        for i in range(1, 1+iterate_num):

            if i % 1000 == 0:
                print 'iteration {}/{}'.format(i, iterate_num)
            output = sess.run([aaa, bbb, op5])
            # print output[0][0]
            # print output[1][0].values
            # for x in output[0]:
            #     print x.shape
            # for x in output[1]:
            #     print x.values.T
            # for x in output[2]:
            #     print x

            if i % break_point == 0:
                current_error = 0
                out = open('error_syn.csv', 'a')
                for _ in range(test_num):
                    output2 = sess.run([test_y, predict_y, cnt, norm])
                    is_right = output2[2][0]
                    if not is_right:
                        current_error += 1
                    # print 'test_y:  ', output2[0],
                    # print '\tpredict_y:  ', output2[1],
                    # print '\terror:  ', current_error,
                print '\tnorm:  ', output2[3]
                print 'error: ', current_error
                #print >> out, current_error
                out.close()
        sess.close()