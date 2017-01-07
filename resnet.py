import tensorflow as tf

class ResNet:
    def __init__(self, n):
        self.inputs = tf.placeholder(tf.float32, shape=[None,32,32,3])
        self.labels = tf.placeholder(tf.float32, shape=[None,10])

        W_conv = tf.Variable(tf.truncated_normal([3,3,3,16], stddev=0.1))
        b_conv = tf.Variable(tf.constant(0.1, shape=[16]))
        l_conv = tf.nn.conv2d(self.inputs, W_conv, strides=[1,1,1,1], padding="SAME")
        l_conv = tf.nn.relu(l_conv + b_conv)

        previous_res = l_conv
        for i in range(n):
            u_res = self.__residual_unit(previous_res, [3,3,16,16])
            previous_res = u_res

        W_downsample1 = tf.Variable(tf.truncated_normal([3,3,16,32], stddev=0.1))
        b_downsample1 = tf.Variable(tf.constant(0.1, shape=[32]))
        l_downsample1 = tf.nn.conv2d(previous_res, W_downsample1, strides=[1,2,2,1], padding="SAME")
        l_downsample1 = tf.nn.relu(l_downsample1 + b_downsample1)

        previous_res = l_downsample1
        for i in range(n):
            u_res = self.__residual_unit(previous_res, [3,3,32,32])
            previous_res = u_res

        W_downsample2 = tf.Variable(tf.truncated_normal([3,3,32,64], stddev=0.1))
        b_downsample2 = tf.Variable(tf.constant(0.1, shape=[64]))
        l_downsample2 = tf.nn.conv2d(previous_res, W_downsample2, strides=[1,2,2,1], padding="SAME")
        l_downsample2 = tf.nn.relu(l_downsample2 + b_downsample2)

        previous_res = l_downsample2
        for i in range(n):
            u_res = self.__residual_unit(previous_res, [3,3,64,64])
            previous_res = u_res

        l_poolf = tf.nn.avg_pool(previous_res, ksize=[1,2,2,1], strides=[1,2,2,1], padding="SAME")
        l_poolf_shape = l_poolf.get_shape()
        l_fc_size = int(l_poolf_shape[1] * l_poolf_shape[2] * l_poolf_shape[3])
        l_pool_flat = tf.reshape(l_poolf, [-1,l_fc_size])

        W_fc1 = tf.Variable(tf.truncated_normal([l_fc_size,512], stddev=0.1))
        b_fc1 = tf.Variable(tf.constant(0.1, shape=[512]))
        l_fc1 = tf.nn.relu(tf.nn.xw_plus_b(l_pool_flat, W_fc1, b_fc1))

        W_fc2 = tf.Variable(tf.truncated_normal([512,10], stddev=0.1))
        b_fc2 = tf.Variable(tf.constant(0.1, shape=[10]))
        l_fc2 = tf.nn.relu(tf.nn.xw_plus_b(l_fc1, W_fc2, b_fc2))

        self.probabilities = tf.nn.softmax(l_fc2)
        self.prediction = tf.argmax(self.probabilities, axis=1)
        self.accuracy = tf.reduce_mean(tf.cast(tf.equal(self.prediction, tf.argmax(self.labels, axis=1)), tf.float32))

        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(l_fc2, self.labels))
        self.train = tf.train.AdamOptimizer(1e-4).minimize(loss)

    def train_model(self, session, inputs, labels):
        session.run(self.train, feed_dict={self.inputs: inputs, self.labels: labels})

    def get_accuracy(self, session, inputs, labels):
        return session.run(self.accuracy, feed_dict={self.inputs: inputs, self.labels: labels})

    def __residual_unit(self, unit_input, filter_shape):
        W_conv1 = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1))
        b_conv1 = tf.Variable(tf.constant(0.1, shape=[filter_shape[3]]))
        l_conv1 = tf.nn.conv2d(unit_input, W_conv1, strides=[1,1,1,1], padding="SAME")
        l_activ1 = tf.nn.relu(l_conv1 + b_conv1)

        W_conv2 = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1))
        b_conv2 = tf.Variable(tf.constant(0.1, shape=[filter_shape[3]]))
        l_conv2 = tf.nn.conv2d(l_activ1, W_conv2, strides=[1,1,1,1], padding="SAME")

        skip_sum = l_conv2 + unit_input
        l_activ2 = tf.nn.relu(skip_sum + b_conv2)

        return l_activ2
