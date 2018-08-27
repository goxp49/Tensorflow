import tensorflow as tf
from tensorflow.python.platform import gfile
from google.protobuf import text_format
import os

# 设置当前项目路径
Project_PATH = os.path.dirname(os.path.realpath(__file__))
# 输出路径
OUTPUT_PATH = os.path.join(Project_PATH, 'inception_v3', 'output')

def convert_pb_to_pbtxt(filename):
    with gfile.FastGFile(filename, 'rb') as f:
        graph_def = tf.GraphDef()

        graph_def.ParseFromString(f.read())

        tf.import_graph_def(graph_def, name='')

        tf.train.write_graph(graph_def, OUTPUT_PATH, 'protobuf.pbtxt', as_text=True)
    return


def convert_pbtxt_to_pb(filename):
    """Returns a `tf.GraphDef` proto representing the data in the given pbtxt file.
    Args:
      filename: The name of a file containing a GraphDef pbtxt (text-formatted
        `tf.GraphDef` protocol buffer data).
    """
    with tf.gfile.FastGFile(filename, 'r') as f:
        graph_def = tf.GraphDef()

        file_content = f.read()

        # Merges the human-readable string in `file_content` into `graph_def`.
        text_format.Merge(file_content, graph_def)
        tf.train.write_graph(graph_def, './', 'protobuf.pb', as_text=False)
    return

if __name__ == "__main__":
    convert_pb_to_pbtxt(os.path.join(OUTPUT_PATH, 'output_graph.pb'))