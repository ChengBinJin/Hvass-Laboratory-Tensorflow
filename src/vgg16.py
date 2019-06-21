########################################################################
#
# The pre-trained VGG16 Model for TensorFlow.
#
# This model seems to produce better-looking images in Style Transfer
# than the Inception 5h model that otherwise works well for DeepDream.
#
# See the Python Notebook for Tutorial #15 for an exapmle usage.
#
# Implemetend in Python 3.5 with TensorFLow 1.13.1
#
########################################################################
#
# This file is part of the TensorFlow Tutorials availabel at:
#
# https://github.com/Hvass-Labs/TensorFlow-Tutorials
#
# Published under the MIT License. See the file LICENSE for details.
#
# Copyright 2016 by Magnus Erik Hvass Pedersen
#
########################################################################

import os
import download
import numpy as np
import tensorflow as tf

########################################################################
#
# Various directories and file-names.
#
# The pre-trained VGG16 model is taken from this tutorials:
# https://github.com/pkmital/CADL/blob/master/session-4/lib/vgg16.py
#
# The class-names are available in the following URL:
# https://s3.amazonaws.com/cadl/models/synset.txt
#
# Internet URL for the file with the VGG16 model.
# Note that this might change in the future and will need to be updated.
data_url = "https://s3.amazonaws.com/cadl/models/vgg16.tfmodel"

# Directory to store the downloaded data.
data_dir = "../vgg16/"

# File containing the TensorFlow graph definition. (Downloaded)
path_graph_def = "vgg16.tfmodel"

########################################################################

def maybe_download():
    # Download the VGG16 model from the internet if it does not already
    # exist in the data_dir. WARNING! The file is about 550 MB.

    print("Downloading VGG16 Model ...")

    # The file on the internet is not stored in a compressed format.
    # This function should not extract the file when it does not have
    # a relevant filename-extensions such as .zip or .tar.gz
    download.maybe_download_and_extract(url=data_url, download_dir=data_dir)

########################################################################


class VGG16:
    # The VGG16 model is a Deep Neural Network which has already been
    # trained for classifying images into 1000 different categories.

    # When you create a new instance of this class, the VGG16 model
    # will be loaded and can be used immediately without training.

    # Name of the tensor for feeding the input image.
    tensor_name_input_image = "images:0"

    # names of the tensors for the dropout random-values..
    tensor_name_dropout = "dropout/random_uniform:0"
    tensor_name_dropout1 = "dropout_1/random_uniform:0"

    # Names for the convolutional layers in the model for use in Style Transfer.
    layer_names = ['conv1_1/conv1_1', 'conv1_2/conv1_2',
                   'conv2_1/conv2_1', 'conv2_2/conv2_2',
                   'conv3_1/conv3_1', 'conv3_2/conv3_2', 'conv3_3/conv3_3',
                   'conv4_1/conv4_1', 'conv4_2/conv4_2', 'conv4_3/conv4_3',
                   'conv5_1/conv5_1', 'conv5_2/conv5_2', 'conv5_3/conv5_3']

    def __init__(self):
        # Now load the model from file. The way TensorFlow
        # does this is confusing and requires several steps.

        # Create a new TensorFlow computational graph.
        self.graph = tf.Graph()

        # Set the new graph as the default.
        with self.graph.as_default():
            # TensorFlow graphs are saved to disk as so-called Protocol Buffers
            # aka. proto-bufs which is a file-format that works on multiple
            # platforms. In this case it is saved as a binary file.

            # Open the graph-def file for binary reading.
            path = os.path.join(data_dir, path_graph_def)
