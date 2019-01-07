import tensorflow as tf


class Cnn(object):

    def __init__(self, conv_layers, fc_layers, filters, learning_rate, beta1, beta2):
        self.__param = {}
        self.__conv_layers = conv_layers
        self.__fc_layers = fc_layers
        self.__filters = filters
        self.__lr = learning_rate
        self.__beta1 = beta1
        self.__beta2 = beta2

    @property
    def x(self):
        return self.__x

    @property
    def y(self):
        return self.__y

    def initialize(self, n_h0, n_w0, n_c0, n_y):
        """
        initialize the w, X, Y
        :param n_h0: height of the image data
        :param n_w0: width of the image data
        :param n_c0: depth of the image data
        :param n_y: number of the labels
        :return:
        """
        f = self.__filters
        convs = self.__conv_layers
        self.__x = tf.placeholder(tf.float32, (None, n_h0, n_w0, n_c0), 'x')
        self.__y = tf.placeholder(tf.float32, (None, n_y), 'y')

        # i->index of layers include pool, temp_i->index of layers exclude pool
        temp_i = 0

        for i in range(len(convs)):
            if i == 0:
                self.__param['w1'] = tf.get_variable('w1', (f[i][0], f[i][0], 3, convs[i]),
                                                     initializer=tf.contrib.layers.xavier_initializer())
                temp_i += 1
            elif convs[i] != 0 and convs[i] != -1:
                self.__param['w' + str(temp_i + 1)] = tf.get_variable('w' + str(temp_i + 1), (
                    f[i][0], f[i][0], self.__param['w' + str(temp_i)].shape[3], convs[i]),
                                                                      initializer=tf.contrib.layers.xavier_initializer())
                temp_i += 1
                # todo: add b
                # self.__param['b' + str(i + 1)] = tf.get_variable('b' + str(i + 1),(1,1,3,convs[i]),initializer= )

    def forward(self, drop_out=False):
        param = self.__param
        f = self.__filters
        convs = self.__conv_layers
        fcs = self.__fc_layers
        a_pre = self.__x

        # i->index of layers include pool, temp_i->index of layers exclude pool
        temp_i = 0

        # convolution forward
        for i in range(len(convs)):
            if convs[i] == 0:
                a = tf.nn.max_pool(a_pre, ksize=[1, f[i][0], f[i][0], 1], strides=[1, f[i][1], f[i][1], 1],
                                   padding=f[i][2])
                a_pre = a
            elif convs[i] == -1:
                a = tf.nn.avg_pool(a_pre, ksize=[1, f[i][0], f[i][0], 1], strides=[1, f[i][1], f[i][1], 1],
                                   padding=f[i][2])
                a_pre = a
            else:
                z = tf.nn.conv2d(a_pre, param['w' + str(temp_i + 1)], strides=[1, f[i][1], f[i][1], 1], padding=f[i][2])
                a = tf.nn.relu(z)
                a_pre = a
                temp_i += 1

        # full connected forward
        a_pre = tf.contrib.layers.flatten(a)
        for i in range(len(fcs)):
            a = tf.contrib.layers.fully_connected(a_pre, fcs[i], activation_fn=None)
            a_pre = a
        self.__a = a

    def cost(self):
        cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.__a, labels=self.__y))
        return cost

    def get_optimizer(self, cost):
        if self.__beta1 and self.__beta2:
            adam = tf.train.AdamOptimizer(self.__lr, beta1=self.__beta1, beta2=self.__beta2).minimize(cost)
        elif self.__beta1:
            adam = tf.train.AdamOptimizer(self.__lr, beta1=self.__beta1).minimize(cost)
        elif self.__beta2:
            adam = tf.train.AdamOptimizer(self.__lr, beta2=self.__beta2).minimize(cost)
        else:
            adam = tf.train.AdamOptimizer(self.__lr).minimize(cost)

        return adam

    def predict(self):
        predict = tf.argmax(self.__a, axis=1)
        accuracy = tf.reduce_mean(tf.cast(tf.equal(predict, tf.argmax(self.__y, axis=1)), "float"))
        return predict, accuracy
