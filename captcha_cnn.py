"""
    通过模块化方式改写验证码训练过程
"""

from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
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
    all_image = os.listdir(TEST_PATH)   ################################
    random_file = random.randint(0, 9)   ################################
    image_file_path = os.path.join(TEST_PATH, all_image[random_file])   ################################
    base = os.path.basename(image_file_path)
    name = os.path.splitext(base)[0]
    image = Image.open(image_file_path)
    image = np.array(image)
    return name, image

# 将名称转接为向量
def name2vec(name):
    vector = np.zeros(MAX_CAPTCHA*CHAR_SET_LEN)
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


# 获得一个权重值
# shape是一个四维矩阵，前两个参数代表卷积核尺寸，第三个代表输入层深度，第四个代表卷积核的个数/过滤器深度/feature map数量
def weight_variable(shape):
    # 正态分布，标准差为0.1，默认最大为1，最小为-1，均值为0
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


# 获得一个偏移项
# shape应与卷积核个数一致
def bias_variable(shape):
    # 创建一个结构为shape矩阵也可以说是数组shape声明其行列，初始化所有值为0.1
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


# 卷积层前向传播算法
# x：输入矩阵
# W：权重
# strides：第一和第四维固定为1，第二和第三维表示卷积核尺寸
def conv2d(x, W):
    # 卷积遍历各方向步数为1，SAME：边缘外自动补0，遍历相乘
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


# 池化层算法
# ksize:池化过滤器的尺寸，第一和第四位固定为1，常用的有[1, 2, 2, 1]和[1, 3, 3, 1]
# strides:池化过滤器的步长，第一和第四位固定为1
def max_pool_2x2(x):
    # 池化卷积结果（conv2d）池化层采用kernel大小为2*2，步数也为2，周围补0，取最大值。数据量缩小了4倍
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


# 生成一个训练batch
def get_next_batch(batch_size=64):
    batch_x = np.zeros([batch_size, IMAGE_HEIGHT*IMAGE_WIDTH])
    batch_y = np.zeros([batch_size, MAX_CAPTCHA*CHAR_SET_LEN])

    for i in range(batch_size):
        name, image = get_name_and_image()
        batch_x[i, :] = 1*(image.flatten())
        batch_y[i, :] = name2vec(name)
    return batch_x, batch_y
# ==========================================    3.定义输入输出结构    ==========================================

def inference():
    ### 第一层卷积操作 ###
    # 将一维图片数据流转换为二维矩阵，第一个参数表示不限订batch数量，最后一个1表示输入层深度
    print('######### 1 ###########')
    x = tf.reshape(X, shape=[-1, IMAGE_HEIGHT, IMAGE_WIDTH, 1])  # 尺寸：114 * 450
    w_conv1 = weight_variable([5, 5, 1, 32])
    b_conv1 = bias_variable([32])
    cout_conv1 = tf.nn.relu(conv2d(x, w_conv1) + b_conv1)
    pout_pool1 = max_pool_2x2(cout_conv1)
    print(cout_conv1)
    print(pout_pool1)
    # dout_dropout1 = tf.nn.dropout(pout_pool1, keep_prob)

    ### 第二层卷积操作 ###
    print('######### 2 ###########')
    w_conv2 = weight_variable([5, 5, 32, 64])
    b_conv2 = bias_variable([64])
    cout_conv2 = tf.nn.relu(conv2d(pout_pool1, w_conv2) + b_conv2)
    pout_pool2 = max_pool_2x2(cout_conv2)
    print(cout_conv2)
    print(pout_pool2)
    # dout_dropout2 = tf.nn.dropout(pout_pool2, keep_prob)

    ### 第三层卷积操作 ###
    print('######### 3 ###########')
    w_conv3 = weight_variable([5, 5, 64, 128])
    b_conv3 = bias_variable([128])
    cout_conv3 = tf.nn.relu(conv2d(pout_pool2, w_conv3) + b_conv3)
    pout_pool3 = max_pool_2x2(cout_conv3)
    print(cout_conv3)
    print(pout_pool3)
    # dout_dropout3 = tf.nn.dropout(pout_pool3, keep_prob)

    ### 第四层全连接操作 ###
    print('######### 4 ###########')
    w_fc1 = weight_variable([15 * 57 * 128, 1024])
    # 1024个偏执数据
    b_fc1 = bias_variable([1024])
    # 将第三层卷积池化结果reshape成只有一行15*57*128个数据# [n_samples, 8, 24, 128] ->> [n_samples, 8*24*128]
    h_pool4_flat = tf.reshape(pout_pool3, [-1, 15 * 57 * 128])
    # 卷积操作，结果是1*1*1024，单行乘以单列等于1*1矩阵，matmul实现最基本的矩阵相乘，不同于tf.nn.conv2d的遍历相乘，自动认为是前行向量后列向量
    out_fc1 = tf.nn.relu(tf.matmul(h_pool4_flat, w_fc1) + b_fc1)
    # 通过dropout层减少过拟合
    out_fc1_drop = tf.nn.dropout(out_fc1, keep_prob)

    ## 第五层输出操作 ##
    print('######### 5 ###########')
    w_fc2 = weight_variable([1024, MAX_CAPTCHA * CHAR_SET_LEN])
    b_fc2 = bias_variable([MAX_CAPTCHA * CHAR_SET_LEN])
    # 最后的分类，结果为1*1*10 softmax和sigmoid都是基于logistic分类算法，一个是多分类一个是二分类
    # y_out = tf.nn.softmax(tf.matmul(out_fc1_drop, w_fc2) + b_fc2)
    y_out = tf.add(tf.matmul(out_fc1_drop, w_fc2), b_fc2)

    return y_out


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
        saver.restore(sess, tf.train.latest_checkpoint('.'))

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
    # train_crack_captcha_cnn()
    crack_captcha()