#########################################################################
#
# The Inception Model 5h for TensorFlow.
#
# This variant of the Inception model is easier to use for DeepDream
# and other imaging techniques. This is because it allows the input
# image to be any size, and the optimized images are also prettier.
#
# It is unclear which Inception model this implements because the
# Google developers have (as usual) neglected to document it.
# It is dubbed the 5h-model because that is the name of the zip-file,
# but it is apparently simpler than the v.3 model.
#
# See the Python Notebook for Tutorial #14 for an examlpe usage.
#
# Implemented in Python 3.5 with TensorFlow version 1.13.1
#
#########################################################################
#
# This file is part of the TensorFlow Tutorials available at:
#
# https://github.com/Hvass-Labs/TensorFlow-Tutorials
#
# Published underthe MIT License. See the file LICENSE for details.
#
# Copyright 2016 by Magnus Erik Hvass Pedersen
#
#########################################################################

import numpy as np
import tensorflow as tf
import download
import os
