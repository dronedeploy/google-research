import argparse
from os import path
import tensorflow as tf
from tensorflow.python.platform import gfile

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--graph", type=str, default="", help="The frozen graph file.")
    args = parser.parse_args()
    with tf.Session() as sess:
        model_filename = args.graph
        with gfile.FastGFile(model_filename, 'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
            g_in = tf.import_graph_def(graph_def)
    logdir = path.dirname(args.graph)
    train_writer = tf.summary.FileWriter(logdir)
    train_writer.add_graph(sess.graph)