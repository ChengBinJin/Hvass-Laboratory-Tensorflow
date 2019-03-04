import os
import math
import time
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.metrics import confusion_matrix
from datetime import timedelta

# Use PrettyTensor to simply Neural Network construction
import prettytensor as pt

print('TensorFlow version: {}'.format(tf.__version__))
