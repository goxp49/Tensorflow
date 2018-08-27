# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse, os

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.image as mpimg # mpimg 用于读取图片

# 设置当前项目路径
Project_PATH = os.path.dirname(os.path.realpath(__file__))
# 测试图集路径
TEST_PATH = os.path.join(Project_PATH, 'inception_v3', 'test_images')
# 输出路径
OUTPUT_PATH = os.path.join(Project_PATH, 'inception_v3', 'output')


# 读取并创建一个图graph来存放Google训练好的模型（函数）
def load_graph(model_file):
    # 使用tf.GraphDef()定义一个空的Graph
    graph = tf.Graph()
    graph_def = tf.GraphDef()

    with open(model_file, "rb") as f:
        graph_def.ParseFromString(f.read())
    with graph.as_default():
        tf.import_graph_def(graph_def)

    return graph


def read_tensor_from_image_file(file_name,
                                input_height=299,
                                input_width=299,
                                input_mean=0,
                                input_std=255):
    input_name = "file_reader"
    output_name = "normalized"
    file_reader = tf.read_file(file_name, input_name)
    if file_name.endswith(".png"):
        image_reader = tf.image.decode_png(
            file_reader, channels=3, name="png_reader")
    elif file_name.endswith(".gif"):
        image_reader = tf.squeeze(
            tf.image.decode_gif(file_reader, name="gif_reader"))
    elif file_name.endswith(".bmp"):
        image_reader = tf.image.decode_bmp(file_reader, name="bmp_reader")
    else:
        image_reader = tf.image.decode_jpeg(
            file_reader, channels=3, name="jpeg_reader")
    float_caster = tf.cast(image_reader, tf.float32)
    dims_expander = tf.expand_dims(float_caster, 0)
    resized = tf.image.resize_bilinear(dims_expander, [input_height, input_width])
    normalized = tf.divide(tf.subtract(resized, [input_mean]), [input_std])
    sess = tf.Session()
    result = sess.run(normalized)

    return result


def load_labels(label_file):
    label = []
    proto_as_ascii_lines = tf.gfile.GFile(label_file).readlines()
    for l in proto_as_ascii_lines:
        label.append(l.rstrip())
    return label


if __name__ == "__main__":
    # file_name = "tensorflow/examples/label_image/data/grace_hopper.jpg"
    file_name = "D:/python_workspace/Tensorflow/inception_v3/test_images/daisy_1.jpg"
    # model_file = "tensorflow/examples/label_image/data/inception_v3_2016_08_28_frozen.pb"
    model_file = os.path.join(OUTPUT_PATH, 'output_graph.pb')
    # label_file = "tensorflow/examples/label_image/data/imagenet_slim_labels.txt"
    label_file = os.path.join(OUTPUT_PATH, 'output_labels.txt')
    # 这里设定的图片处理后的尺寸，在'read_tensor_from_image_file'中会依据该数值resize处理输入图片尺寸
    input_height = 224
    input_width = 224
    input_mean = 0
    input_std = 255
    # input_layer = "input"
    input_layer = "Placeholder"
    # output_layer = "InceptionV3/Predictions/Reshape_1"
    output_layer = "final_result"

    graph = load_graph(model_file)
    input_name = "import/" + input_layer
    output_name = "import/" + output_layer
    input_operation = graph.get_operation_by_name(input_name)
    output_operation = graph.get_operation_by_name(output_name)

# ————————————————————————————————————————
    # 遍历目录
    for root, dirs, files in os.walk(TEST_PATH):
        for file_name in files:
            # 打印图片路径及名称
            image_path = os.path.join(root, file_name)
            print(image_path)
            # 显示图片
            img = mpimg.imread(image_path)
            # img = Image.open(image_path)
            plt.imshow(img)
            plt.axis('off')
            plt.show()
# ————————————————————————————————————————
            t = read_tensor_from_image_file(
                image_path,
                input_height=input_height,
                input_width=input_width,
                input_mean=input_mean,
                input_std=input_std)

            with tf.Session(graph=graph) as sess:
                results = sess.run(output_operation.outputs[0], {
                    input_operation.outputs[0]: t
                })
            results = np.squeeze(results)

            top_k = results.argsort()[-5:][::-1]
            labels = load_labels(label_file)
            for i in top_k:
                print(labels[i], results[i])
