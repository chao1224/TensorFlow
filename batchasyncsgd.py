import tensorflow as tf
import os
import datetime

tf.app.flags.DEFINE_integer("task_index", 0, "Index of the worker task")
FLAGS = tf.app.flags.FLAGS

worker_num = 5
num_features = 33762578
batch_size = 100

# iteration num, should be 2e7
iterate_num = int(1e6 / batch_size)
# test num is test size, should be 1e4
test_num = int(2e3)
# break point is how often we run a test, should be 1e5
break_point = int(1e4 / batch_size)

g = tf.Graph()

input_producers = [
    ["./data/tfrecords00", "./data/tfrecords01", "./data/tfrecords02", "./data/tfrecords03", "./data/tfrecords04"],
    ["./data/tfrecords05", "./data/tfrecords06", "./data/tfrecords07", "./data/tfrecords08", "./data/tfrecords09"],
    ["./data/tfrecords10", "./data/tfrecords11", "./data/tfrecords12", "./data/tfrecords13", "./data/tfrecords14"],
    ["./data/tfrecords15", "./data/tfrecords16", "./data/tfrecords17", "./data/tfrecords18", "./data/tfrecords19"],
    ["./data/tfrecords20", "./data/tfrecords21"],
    ["./data/tfrecords23"]
]

with g.as_default():

    # creating a model variable on task 0. This is a process running on node vm-48-1
    with tf.device("/job:worker/task:0"):
        w = tf.Variable(tf.zeros([num_features]), name="model")
        params = tf.Variable(tf.zeros([num_features]))

    def read_my_file_format(filename_queue):
        reader = tf.TFRecordReader()
        _, serialized_example = reader.read(filename_queue)
        return serialized_example

    # 1. update indice to parameter server
    with tf.device("/job:worker/task:%d" % FLAGS.task_index):
        filename_queue = tf.train.string_input_producer(input_producers[FLAGS.task_index], num_epochs=None)
        serialized_example = read_my_file_format(filename_queue)
        min_after_dequeue = 10000
        capacity = min_after_dequeue + 10 * batch_size
        serialized_example_batch = tf.train.shuffle_batch(
            [serialized_example], batch_size=batch_size, capacity=capacity,
            min_after_dequeue=min_after_dequeue)
        features = tf.parse_example(serialized_example_batch,
                                    features={
                                        'label': tf.FixedLenFeature([1], dtype=tf.int64),
                                        'index': tf.VarLenFeature(dtype=tf.int64),
                                        'value': tf.VarLenFeature(dtype=tf.float32),
                                    })
        label = features['label']
        index = features['index']
        value = features['value']

    # 2. push dense weights to all workers
    with tf.device("/job:worker/task:0"):
        weight = tf.gather(w, index.values)

    # 3. calculate gradients
    with tf.device("/job:worker/task:%d" % FLAGS.task_index):
        y = tf.cast(label, tf.float32)[0]
        x = value.values
        a = tf.mul(weight, x)
        b = tf.reduce_sum(a)
        c = tf.sigmoid(tf.mul(y, b))
        d = tf.mul(y, c-1)
        local_gradient = tf.mul(d, x)

    # 4. update gradients
    with tf.device("/job:worker/task:0"):
        ggg = local_gradient
        w = tf.scatter_sub(w, index.values, local_gradient)

    # 5. assign value for params
    with tf.device("/job:worker/task:0"):
        update_params = tf.assign(params, w)

    # 6. calculate test error
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

        sparse_params = tf.gather(params, index.values)
        test_x = value.values
        test_y = tf.cast(label, tf.float32)[0]

        predict_confidence = tf.reduce_sum(tf.mul(sparse_params, test_x))
        predict_y = tf.sign(predict_confidence)
        cnt = tf.equal(test_y, predict_y)
        norm = tf.reduce_sum(tf.mul(sparse_params, sparse_params))


    with tf.Session("grpc://vm-22-%d:2222" % (FLAGS.task_index+1)) as sess:
        print datetime.datetime.now()
        if FLAGS.task_index == 0:
            sess.run(tf.initialize_all_variables())
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        for i in range(1, 1+iterate_num):
            output = sess.run([ggg])

            if i % break_point == 0:
                print 'reach break point'
                sess.run(update_params)
                current_error = 0
                break_point_params = w.eval()
                out = open('error_batched_asyn.csv', 'a')
                for j in range(test_num):
                    output2 = sess.run([test_y, predict_y, cnt, norm])
                    is_right = output2[2]
                    if not is_right:
                        current_error += 1
                print 'current error:  ', current_error
                out.close()
        print datetime.datetime.now()
        sess.close()