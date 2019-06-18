# TensorFlow Tutorial #14
# DeepDream
# Introduction
# Flowchart
# Recursive Optimization

# Imports
import math
import random
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

# Image manipulation
import PIL.Image
from scipy.ndimage.filters import gaussian_filter

print("TensorFlow version: {}".format(tf.__version__))

# Inception Model
# import inception5h
# inception.data_dir = "../inception/5h/"

# inception5h.maybe_download()
# model = indeption5h.Inception5h()

# print("len(model.layer_tensors): {}".format(len(model.layer_tensors)))
