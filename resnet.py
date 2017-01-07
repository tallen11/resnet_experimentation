import tensorflow as tf

class ResNet:
    def __init__(self, res_unit_count):
        self.inputs = tf.placeholder(tf.float32, shape=[None,32,32,3])
        self.labels = tf.placeholder(tf.float32, shape=[None,10])
        self.dropout = tf.placeholder(tf.float32)

        W_conv1 = tf.Variable(tf.truncated_normal([5,5,3,32], stddev=0.1))
        b_conv1 = tf.Variable(tf.constant(0.1, shape=[32]))
        l_conv1 = tf.nn.conv2d(self.inputs, W_conv1, strides=[1,1,1,1], padding="SAME")
        l_activ1 = tf.nn.relu(l_conv1 + b_conv1)
        l_pool1 = tf.nn.max_pool(l_activ1, ksize=[1,2,2,1], strides=[1,2,2,1], padding="SAME")

        u_res1, u_out1 = self.__residual_unit(l_pool1, l_pool1, [3,3,32,32])
        prev_ouput = u_res1
        prev_concat = u_out1
        for i in range(res_unit_count - 1):
            u_res, u_out = self.__residual_unit(prev_concat, prev_ouput, [3,3,64,32], halve_filters=True)
            prev_ouput = u_res
            prev_concat = u_out

        l_poolf = tf.nn.max_pool(prev_concat, ksize=[1,2,2,1], strides=[1,2,2,1], padding="SAME")
        l_poolf_shape = l_poolf.get_shape()
        l_fc_size = int(l_poolf_shape[1] * l_poolf_shape[2] * l_poolf_shape[3])

        W_fc1 = tf.Variable(tf.truncated_normal([l_fc_size,1024], stddev=0.1))
        b_fc1 = tf.Variable(tf.constant(0.1, shape=[1024]))
        l_fc1 = tf.reshape(l_poolf, [-1,l_fc_size])
        l_fc1 = tf.nn.relu(tf.nn.xw_plus_b(l_fc1, W_fc1, b_fc1))

        l_dropout = tf.nn.dropout(l_fc1, self.dropout)

        W_fc_s = tf.Variable(tf.truncated_normal([1024,10], stddev=0.1))
        b_fc_s = tf.Variable(tf.constant(0.1, shape=[10]))
        l_fc_s = tf.nn.xw_plus_b(l_dropout, W_fc_s, b_fc_s)

        self.probabilities = tf.nn.softmax(l_fc_s)
        self.prediction = tf.argmax(self.probabilities, axis=1)
        self.accuracy = tf.reduce_mean(tf.cast(tf.equal(self.prediction, tf.argmax(self.labels, axis=1)), tf.float32))

        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(l_fc_s, self.labels))
        self.train = tf.train.AdamOptimizer(1e-4).minimize(loss)

    def train_model(self, session, inputs, labels):
        session.run(self.train, feed_dict={self.inputs: inputs, self.labels: labels, self.dropout: 0.5})

    def get_accuracy(self, session, inputs, labels):
        return session.run(self.accuracy, feed_dict={self.inputs: inputs, self.labels: labels, self.dropout: 1.0})

    def __residual_unit(self, unit_input, prev_ouput, filter_shape, halve_filters=False):
        W_conv1 = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1))
        b_conv1 = tf.Variable(tf.constant(0.1, shape=[filter_shape[3]]))
        l_conv1 = tf.nn.conv2d(unit_input, W_conv1, strides=[1,1,1,1], padding="SAME")
        l_activ1 = tf.nn.relu(l_conv1 + b_conv1)

        if halve_filters:
            filter_shape[2] = filter_shape[2] // 2
        W_conv2 = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1))
        b_conv2 = tf.Variable(tf.constant(0.1, shape=[filter_shape[3]]))
        l_conv2 = tf.nn.conv2d(l_activ1, W_conv2, strides=[1,1,1,1], padding="SAME")
        l_activ2 = tf.nn.relu(l_conv2 + b_conv2)

        return l_activ2, tf.concat(3, [l_activ2, prev_ouput])
