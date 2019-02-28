import os
import math
import time
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.metrics import confusion_matrix
from datetime import timedelta
from mnist import MNIST

# Use PrettyTensor to simplify Neural Network construction.
# import prettytensor as pt

print("Tensorflow version: {0:6}".format(tf.__version__))

# Load Data
data = MNIST(data_dir="../data/MNIST")

print("Size of :")
print("- Training-set:\t\t{}".format(data.num_train))
print("- Validation-set:\t{}".format(data.num_val))
print("- Test-set:\t\t{}".format(data.num_test))

# Class numbers
# data.test.cls = np.argmax(data.test.labels, axis=1)
# data.validation.cls = np.argmax(data.validation.labels, axis=1)

# Helper-function for creating random training-sets
combined_images = np.concatenate([data.x_train, data.x_val], axis=0)
combined_labels = np.concatenate([data.y_train, data.y_val], axis=0)

print("combined_images shape: {0}".format(combined_images.shape))
print("combined_labels shape: {0}".format(combined_labels.shape))
print("combined_images dtype: {0}".format(combined_images.dtype))
print("combined_labels dtype: {0}".format(combined_labels.dtype))

combined_size = len(combined_images)
print("combined_size: {0:6}".format(combined_size))

train_size = int(0.8 * combined_size)
print("train_size: {0}".format(train_size))

validation_size = combined_size - train_size
print("validation_size: {}".format(validation_size))

def random_training_set():
    # Create a randomized index into the full / combined training-set.
    idx = np.random.permutation(combined_size)

    # Split the random index into trainining- and validation-sets.
    idx_train = idx[0:train_size]
    idx_validation = idx[train_size:]

    # Select the images and labels for the new training-set.
    x_train = combined_images[idx_train, :]
    y_train = combined_labels[idx_train, :]

    # Select the images and labels for the new validation-set.
    x_validation = combined_images[idx_validation, :]
    y_validation = combined_labels[idx_validation, :]

    # Return the new training- and validation-sets.
    return x_train, y_train, x_validation, y_validation

# Data Dimensions
# We know that MNIST images are 28 pixels in each dimension.
img_size = 28

# Image are stored in one-dimensional arrays of this length.
img_size_flat = img_size * img_size

# Tuple with height and width of iamges used to reshape arrays.
img_shape = (img_size, img_size)

# Number of colour channels for the images: 1 channel for gray-scale.
num_channels = 1

# Number of classes, one class for each of 10 digits.
num_classes = 10

# Helper-function for plotting images
def plot_images(images_,                     # Image to plot, 2-d array.
                cls_true_,                   # True class-no for images.
                ensemble_cls_pred=None,     # Ensemble predicted class-no.
                best_cls_pred=None):        # Best-net predicted class-no.

    assert len(images_) == len(cls_true_)

    # Create figure with 3x3 sub-plots.
    fig, axes = plt.subplots(3, 3)

    # Adjust vertical spacing if we need to pring ensemble and best-net.
    if ensemble_cls_pred is None:
        hspace = 0.3
    else:
        hspace = 1.0
    fig.subplots_adjust(hspace=hspace, wspace=0.3)

    # For each of the sub-plots.
    for i, ax in enumerate(axes.flat):
        # There may not be enoguh images for all sub-plots.
        if i < len(images_):
            # Plot image.
            ax.imshow(images_[i].reshape(img_shape), cmap='binary')

            # Show true and predicted classes.
            if ensemble_cls_pred is None:
                xlabel = "True: {0}".format(cls_true_[i])
            else:
                msg = "True: {0}\nEnsemble: {1}\nBest Net: {2}"
                xlabel = msg.format(cls_true_[i],
                                    ensemble_cls_pred[i],
                                    best_cls_pred[i])

            # Show the classes as the label on the x-axis.
            ax.set_xlabel(xlabel)

        # Remove ticks from the plot.
        ax.set_xticks([])
        ax.set_yticks([])

    # Ensure the plot is shown correctly with multiple plots
    # in a single Notebook cell.
    plt.show()

# Plot a few images to see if data is correct
# Get the first iamges from the test-set.
images = data.x_test[0:9]

# Get the true classes for those images.
cls_true = data.y_test_cls[0:9]

# Plot the images and labels using our helper-function above.
plot_images(images_=images, cls_true_=cls_true)

# TensorFlow Graph
x = tf.placeholder(tf.float32, shape=[None, img_size_flat], name='x')
x_image = tf.reshape(x, [-1, img_size, img_size, num_channels])
y_true = tf.placeholder(tf.float32, shape=[None, num_classes], name='y_true')
y_true_cls = tf.argmax(y_true, dimension=1)

# Neural Network
# Helper-functions for creating new variables
def new_weights(shape):
    return tf.Variable(tf.truncated_normal(shape, stddev=0.05))

def new_biases(length):
    return tf.Variable(tf.constant(0.05, shape=[length]))

# Helper-function for creating a new Convolutional Layer
def new_conv_layer(inputs,                  # The previous layer.
                   num_input_channels,      # Num. channels in prev. layer.
                   filter_size,             # Width and height of each filter.
                   num_filters,             # Number of filters.
                   use_pooling=True):       # Use 2x2 max-pooling.

    # Shape of the filter-weights for the convolution.
    # This format is determined by the TensorFlow API.
    shape = [filter_size, filter_size, num_input_channels, num_filters]

    # Create new weights aka. filters with the given shape.
    weights = new_weights(shape=shape)

    # Create new biases, one for each filter.
    biases = new_biases(length=num_filters)

    # Create the TensorFlow operation for convolution.
    # Note teh strides ar eset to 1 in all dimensions.
    # The first and last stride must always be 1,
    # because the first is for the image-number and
    # the last is for the input-channel.
    # But e.g. strides=[1, 2, 2, 1] would mean that the filter
    # is moved 2 pixels across the x- and y-axis fo the image.
    # The padding is set to 'SAME' which means the input iamge
    # is padded with zeroes so the size of the output is the same.
    layer = tf.nn.conv2d(input=inputs,
                         filter=weights,
                         strides=[1, 1, 1, 1],
                         padding='SAME')

    # Add the biases to the results ofthe convolution.
    # A bias-value is added to each filter-channel.
    layer += biases

    # Use pooling to down-sample the iamge resolution?
    if use_pooling:
        # This is 2x2 max-poooling, which means that we
        # conside 2x2 windows and select the largest value
        # in each window. Then we move 2 pixels to the next window.
        layer = tf.nn.max_pool(value=layer,
                               ksize=[1, 2, 2, 1],
                               strides=[1, 2, 2, 1],
                               padding='SAME')

    #
