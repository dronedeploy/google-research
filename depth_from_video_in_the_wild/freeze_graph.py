import os, argparse
import logging
import tensorflow as tf

# The original freeze_graph function
# from tensorflow.python.tools.freeze_graph import freeze_graph 

dir = os.path.dirname(os.path.realpath(__file__))

def freeze_graph(model_dir, input_meta_file):
    """Extract the sub graph defined by the output nodes and convert 
    all its variables into constant 

    Args:
        model_dir: the root folder containing the checkpoint state file        
    """
    if not tf.gfile.Exists(model_dir):
        raise AssertionError(
            "Export directory doesn't exists. Please specify an export "
            "directory: %s" % model_dir)

    # We precise the file fullname of our freezed graph
    output_graph = os.path.join(model_dir, "frozen_model.pb")
    output_node_names = ['disp1']

    # TensorFlow lazily loads contrib modules
    tf.contrib.resampler

    # We clear devices to allow TensorFlow to control on which device it will load operations
    clear_devices = True    
    # We start a session using a temporary fresh Graph
    with tf.Session(graph=tf.Graph()) as sess:
        saver = tf.train.import_meta_graph(input_meta_file)
        saver.restore(sess, input_meta_file[:-5])
        # We use a built-in TF helper to export variables to constants
        output_graph_def = tf.graph_util.convert_variables_to_constants(
            sess, # The session is used to retrieve the weights
            tf.get_default_graph().as_graph_def(), # The graph_def is used to retrieve the nodes 
            output_node_names # The output node names are used to select the useful nodes
        ) 

        # Finally we serialize and dump the output graph to the filesystem
        with tf.gfile.GFile(output_graph, "wb") as f:
            f.write(output_graph_def.SerializeToString())
        print("%d ops in the final graph." % len(output_graph_def.node))

    return output_graph_def

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", type=str, default="", help="Model folder to export")
    parser.add_argument("--meta_file", type=str, default="", help="Checkpoint to load (.meta file)")    
    args = parser.parse_args()

    freeze_graph(args.model_dir, args.meta_file)