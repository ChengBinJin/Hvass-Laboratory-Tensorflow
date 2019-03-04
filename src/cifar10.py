######################################################################################################
# Functions for donwloading the CIFAR-10 data-set from the internet and loading it into memory.
#
# Implemented in Python 3.6
#
# Usage:
# 1) Set the variable data_path with the desired storage path.
# 2) Call maybe_download_and_extract() to download the data-set if it is not already located in the
#    given data_path.
# 3) Call load_class_names() to get an array of the class-names.
# 4) Call load_training_data() and load_test_data() to get
#    the images, class-numbers and one-hot encoded class-labels for the training-set and test-set.
# 5) Use the returned data in your own program.
#
# Format:
# The images for training- and test-sets are returned as 4-dim numpy arrays each with the shape:
# [image_number, height, width, channel] where the individual pixels are floats between 0.0 and 1.0.
#
######################################################################################################
#
# This file is part of the TensorFlow Tutorials available at:
#
# https://github.com/Hvass-Labs/TensorFlow-Tutorials
#
# Published under the MIT License. See the file LICENSE for details.
#
# Copyright 2016 by Magnus Erik Hvass Pedersen
#
######################################################################################################

import os
import pickle
import numpy as np

import download
from dataset import one_hot_encoded

######################################################################################################
# Directory where you want to download and save the data-set.
# Set this before you start calling any of the functions below.
data_path = "../data/CIFAR-10/"

# URL for the data-set on the internet.
data_url = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
######################################################################################################
# Various constants for the size of the images.
# Use these constants in your own program.

# Width and height of each image.
img_size = 32

# Number of channels in each image, 3 channels: Red, Green, Blue.
num_channels = 3

# Length of an image when flattened to a 1-dim array.
img_size_flat = img_size * img_size * num_channels

# Number of classes.
num_classes = 10

######################################################################################################
