"""
    通过CNN网络训练与测速MNIST手写数据集

"""

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.examples.tutorials.mnist import mnist
import tensorflow.contrib.slim as slim

# ==========================================    1.常量设定    ==========================================
# 图片尺寸
IMAGE_SIZE = 28

# ==========================================    2.参数设定    ==========================================

mnist = input_data.read_data_sets('./MNIST/', one_hot=True)

X = tf.placeholder("float", shape=[None, 784])
Y_ = tf.placeholder("float", shape=[None, 10])

keep_prob=tf.placeholder('float')

# ==========================================    3.函数声明    ==========================================
def inference():
    # 将输入数据转换为四维矩阵
    x_image = tf.reshape(X, [-1, IMAGE_SIZE, IMAGE_SIZE, 1])  # shape of x is [N,28,28,1]

    # 第一层卷积
    net = slim.conv2d(x_image, 32, [5, 5], scope='conv1')  # shape of net is [N,28,28,32]
    net = slim.max_pool2d(net, [2, 2], scope='conv1')  # shape of net is [N,14,14,32]

    # 第二层卷积
    net = slim.conv2d(net, 64, [5, 5], scope='conv2')  # shape of net is [N,14,14,64]
    net = slim.max_pool2d(net, [2, 2], scope='conv2')  # shape of net is [N,7,7,64]

    # 第三层全连接
    net = tf.reshape(net, [-1, 7 * 7 * 64])  # [N,7*7*64]
    net = slim.fully_connected(net,1024,scope='fc1')    #shape of net is [N,1024],1024为自定义输出节点数量
    net = tf.nn.dropout(net, keep_prob)

    # 第四层全连接
    net = slim.fully_connected(net, 10, scope='fc1')
    net = tf.nn.dropout(net, keep_prob)

    return net


def train_mnist():
    net_output = inference()
    # 可以先计算 loss再通过‘get_total_loss’获得全部的loss
    slim.losses.sparse_softmax_cross_entropy(net_output, Y_)
    loss = slim.losses.get_total_loss()
    optimizer = tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss)

    correct_prediction = tf.equal(tf.argmax(net_output, axis=1), tf.argmax(Y_, axis=1))  # shape of correct_prediction is [N]
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, 'float'))
    # 初始化所有变量
    init = tf.global_variables_initializer()

    with tf.Session() as sess:
        sess.run(init)
        for i in range(10000):
            batch = mnist.train.next_batch(50)
            if i % 100 == 0:
                train_accuracy = sess.run(accuracy, feed_dict={X: batch[0], Y_: batch[1], keep_prob: 1.0})
                print('step %d,training accuracy  %g !!!!!!!' % (i, train_accuracy))
            sess.run(optimizer, feed_dict={X: batch[0], Y_: batch[1], keep_prob: 0.5})

        total_accuracy = sess.run(accuracy, feed_dict={X: mnist.test.images, Y_: mnist.test.labels, keep_prob: 1.0})
        print('test_accuracy  %s!!!!!!!' % (total_accuracy))