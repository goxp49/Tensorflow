"""
    通过模块化方式改写验证码训练过程
"""

import tensorflow as tf
import tensorflow.contrib.slim as slim
from PIL import Image
import numpy as np
import os
import random

# 设置TF的Log输出等级，默认为1；2只输入警告和错误；3只输出错误
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# 设置当前项目路径
Project_PATH = os.path.dirname(os.path.realpath(__file__))
# 训练图集路径
TRAIN_PATH = os.path.join(Project_PATH, 'captcha4')
# 测试图集路径
TEST_PATH = os.path.join(Project_PATH, 'captcha5')

# ==========================================    1.参数设定    ==========================================
# 验证码尺寸
IMAGE_HEIGHT = 114
IMAGE_WIDTH = 450
# 每大验证码数量
MAX_CAPTCHA = 6
# 验证码每一位有几种可能的值
CHAR_SET_LEN = 26

X = tf.placeholder(tf.float32, [None, IMAGE_HEIGHT * IMAGE_WIDTH])
Y = tf.placeholder(tf.float32, [None, MAX_CAPTCHA * CHAR_SET_LEN])
# dropout参数，0 ~ 1,表示舍弃节点的概率，为1时不起作用
keep_prob = tf.placeholder(tf.float32)


# ==========================================    2.函数声明    ==========================================
# 获得测试图像的名称和数据流
def get_name_and_image():
    all_image = os.listdir(TRAIN_PATH)
    # all_image = os.listdir(TEST_PATH)
    random_file = random.randint(0, 3429)
    # random_file = random.randint(0, 9)
    image_file_path = os.path.join(TRAIN_PATH, all_image[random_file])
    # image_file_path = os.path.join(TEST_PATH, all_image[random_file])
    base = os.path.basename(image_file_path)
    name = os.path.splitext(base)[0]
    image = Image.open(image_file_path)
    image = np.array(image)
    return name, image


# 将名称转接为向量
def name2vec(name):
    vector = np.zeros(MAX_CAPTCHA * CHAR_SET_LEN)
    for i, c in enumerate(name):
        idx = i * 26 + ord(c) - 97
        vector[idx] = 1
    return vector


# 将向量转换为名称
def vec2name(vec):
    name = []
    for i in vec:
        a = chr(i + 97)
        name.append(a)
    return "".join(name)


# 生成一个训练batch
def get_next_batch(batch_size=64):
    batch_x = np.zeros([batch_size, IMAGE_HEIGHT * IMAGE_WIDTH])
    batch_y = np.zeros([batch_size, MAX_CAPTCHA * CHAR_SET_LEN])

    for i in range(batch_size):
        name, image = get_name_and_image()
        batch_x[i, :] = 1 * (image.flatten())
        batch_y[i, :] = name2vec(name)
    return batch_x, batch_y


# ==========================================    3.定义输入输出结构    ==========================================

def inference():
    ### 第一层卷积操作 ###
    # 将一维图片数据流转换为二维矩阵，第一个参数表示不限订batch数量，最后一个1表示输入层深度
    print('######### 1 ###########')
    x_image = tf.reshape(X, shape=[-1, IMAGE_HEIGHT, IMAGE_WIDTH, 1])  # 尺寸：114 * 450
    net = slim.conv2d(x_image, 16, [3, 3])  # shape of net is [N,114,450,32]
    net = slim.conv2d(net, 16, [3, 3])  # shape of net is [N,114,450,32]
    net = slim.max_pool2d(net, [2, 2])  # shape of net is [N,57,225,32]

    ### 第二层卷积操作 ###
    print('######### 2 ###########')
    net = slim.conv2d(net, 32, [3, 3])
    net = slim.conv2d(net, 32, [3, 3])
    net = slim.max_pool2d(net, [2, 2])  # shape of net is [N,29,113,64]

    ### 第三层卷积操作 ###
    print('######### 3 ###########')
    net = slim.conv2d(net, 64, [3, 3])
    net = slim.conv2d(net, 64, [3, 3])
    net = slim.max_pool2d(net, [2, 2])  # shape of net is [N,15,57,128]
    net = tf.nn.dropout(net, keep_prob)

    ### 第四层全连接操作 ###
    print('######### 4 ###########')
    net = tf.reshape(net, [-1, 14 * 56 * 64])
    net = slim.fully_connected(net, 1024, scope='fc1')  # shape of net is [N,1024],1024为自定义输出节点数量
    net = tf.nn.dropout(net, keep_prob)

    ## 第五层输出操作 ##
    print('######### 5 ###########')
    net = slim.fully_connected(net, MAX_CAPTCHA * CHAR_SET_LEN, scope='fc2')
    return net


def train_crack_captcha_cnn(learn_ratio=0.001):
    # 获得前向传播结果
    y_out = inference()
    # 定义loss(最小误差概率)，选定优化优化loss，
    loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=y_out, labels=Y))
    optimizer = tf.train.AdamOptimizer(learn_ratio).minimize(loss)  # 调用优化器优化，其实就是通过喂数据争取cross_entropy最小化

    # 计算出准确率
    predict = tf.reshape(y_out, [-1, MAX_CAPTCHA, CHAR_SET_LEN])
    max_idx_out = tf.argmax(predict, 2)
    max_idx_label = tf.argmax(tf.reshape(Y, [-1, MAX_CAPTCHA, CHAR_SET_LEN]), 2)
    correct_pred = tf.equal(max_idx_out, max_idx_label)
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    saver = tf.train.Saver()
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        # 恢复之前数据
        try:
            saver.restore(sess, tf.train.latest_checkpoint('.'))
        except ValueError:
            print('没有可用于恢复的数据')

        step = 0
        while True:
            batch_x, batch_y = get_next_batch(64)
            _, loss_ = sess.run([optimizer, loss], feed_dict={X: batch_x, Y: batch_y, keep_prob: 0.5})
            print(step, loss_)

            # 每100 step计算一次准确率
            if step % 100 == 0:
                batch_x_test, batch_y_test = get_next_batch(100)
                acc = sess.run(accuracy, feed_dict={X: batch_x_test, Y: batch_y_test, keep_prob: 1.})
                print('当前阶段为：%s，准确率为：%s' % (step, acc))
                # 如果准确率大于50%,保存模型,完成训练
                if acc > 0.99:
                    saver.save(sess, "./crack_capcha.model", global_step=step)
                    break

            step += 1


def crack_captcha():
    # CNN最终输出的特征向量
    output = inference()

    saver = tf.train.Saver()
    with tf.Session() as sess:
        saver.restore(sess, tf.train.latest_checkpoint('.'))
        n = 1
        while n <= 10:
            text, image = get_name_and_image()
            image = 1 * (image.flatten())
            predict = tf.argmax(tf.reshape(output, [-1, MAX_CAPTCHA, CHAR_SET_LEN]), 2)
            text_list = sess.run(predict, feed_dict={X: [image], keep_prob: 1})
            vec = text_list[0].tolist()
            predict_text = vec2name(vec)
            print("正确: {}  预测: {}".format(text, predict_text))
            n += 1


if __name__ == '__main__':
    train_crack_captcha_cnn()
    # crack_captcha()
