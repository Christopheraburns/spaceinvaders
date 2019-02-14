import tensorflow as tf


class DQNetwork:
    def __init__(self, state_size, action_size, learning_rate, name='DQNetwork'):
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate

        try:
            with tf.variable_scope(name):
                self.inputs_ = tf.placeholder(tf.float32, [None, *state_size], name="inputs")
                self.actions_ = tf.placeholder(tf.float32, [None, self.action_size], name="actions_")

                # Target_Q = R(s,a) + ymax Qhat(s', a')
                self.target_Q = tf.placeholder(tf.float32, [None], name="target")

                # First convolution
                # ELU - https://ml-cheatsheet.readthedocs.io/en/latest/activation_functions.html#elu
                # input is 110x84x4 (Frame size from Space Invaders)
                self.conv1 = tf.layers.conv2d(inputs=self.inputs_,
                                              filters=32,
                                              kernel_size=[8, 8],
                                              strides=[4, 4],
                                              padding="VALID",
                                              kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                                              name="conv1")

                # capture the output of the conv1 layer
                self.conv1_out = tf.nn.elu(self.conv1, name="conv1_out")

                # Second Convolution:
                # ELU - https://ml-cheatsheet.readthedocs.io/en/latest/activation_functions.html#elu
                # input of Layer2 is the output of layer 1
                self.conv2 = tf.layers.conv2d(inputs=self.conv1_out,
                                              filters=64,
                                              kernel_size=[4, 4],
                                              strides=[2, 2],
                                              padding="VALID",
                                              kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                                              name="conv2")
                # capture the output of layer 2
                self.conv2_out = tf.nn.elu(self.conv2, name="conv2_out")

                # Third Convolution:
                # ELU - https://ml-cheatsheet.readthedocs.io/en/latest/activation_functions.html#elu
                # Input of Layer3 is the output of layer 2
                self.conv3 = tf.layers.conv2d(inputs=self.conv2_out,
                                              filters=64,
                                              kernel_size=[3, 3],
                                              strides=[2, 2],
                                              padding="VALID",
                                              kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                                              name="conv3")

                self.conv3_out = tf.nn.elu(self.conv3, name="conv3_out")

                self.flatten = tf.contrib.layers.flatten(self.conv3_out)

                self.fc = tf.layers.dense(inputs=self.flatten,
                                          units=512,
                                          activation=tf.nn.elu,
                                          kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                          name="fc1")

                self.output = tf.layers.dense(inputs=self.fc,
                                              kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                              units=self.action_size,
                                              activation=None)

                # Q is our predicted Q Value
                self.Q = tf.reduce_sum(tf.multiply(self.output, self.actions_))

                # The loss is the difference between our predicted Q value and the Q target
                # sum (Q-target - Q)^2
                self.loss = tf.reduce_mean(tf.square(self.target_Q - self.Q))

                self.optimizer = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)
        except Exception as err:
            print("Error occured in the DQNetwork class: {}".format(err))
