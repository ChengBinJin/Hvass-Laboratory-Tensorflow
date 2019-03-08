import os
import math
import time
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.metrics import confusion_matrix
from datetime import timedelta
from tensorflow.python.training import moving_averages

import cifar10
from cifar10 import img_size, num_channels, num_classes

# Use PrettyTensor to simply Neural Network construction
# import prettytensor as pt

print('TensorFlow version: {}'.format(tf.__version__))

# Configuration of Neural Network
# Convolutional Layer 1
filter_size1 = 5        # Convolution filters are 5 x 5 pixels.
num_filters1 = 64       # There are 64 of these filters.

filter_size2 = 5        # Convolution filters are 5 x 5 pixels.
num_filters2 = 64        # There are 64 of these filters.

fc_size1 = 256          # Number of neurons in fully-connected layer
fc_size2 = 128          # Number of neurons in fully-connected layer

# Load Data
cifar10.maybe_download_and_extract()

class_names = cifar10.load_class_names()
print('CIFAR-10 class names: {}'.format(class_names))

images_train, cls_train, labels_train = cifar10.load_training_data()
images_test, cls_test, labels_test = cifar10.load_test_data()

print("Size of:")
print("- Training-set:\t\t{}".format(len(images_train)))
print("- Test-set:\t\t{}".format(len(images_test)))

# Data Dimensions
img_size_cropped = 24

# Helper-function for plotting images
def plot_images(images_, cls_true_, cls_pred=None, smooth=True):
    assert len(images_) == len(cls_true_) == 9

    # Create figure with sub-plots.
    fig, axes = plt.subplots(3, 3)

    # Adjust vertical spacing if we need to print ensemble and best-net.
    if cls_pred  is None:
        hspace = 0.3
    else:
        hspace=0.6
    fig.subplots_adjust(hspace=hspace, wspace=0.3)

    for i, ax in enumerate(axes.flat):
        # Interpolation type.
        if smooth:
            interpolation = 'spline16'
        else:
            interpolation = 'nearest'

        # Plot image.
        ax.imshow(images_[i, :, :, :], interpolation=interpolation)

        # Nmae of the true class.
        cls_true_name = class_names[cls_true_[i]]

        # Show true and predicted classes.
        if cls_pred is None:
            xlabel = "True: {0}".format(cls_true_name)
        else:
            # Name of the predicted class.
            cls_pred_name = class_names[cls_pred[i]]

            xlabel = "True: {0}\nPred: {1}".format(cls_true_name, cls_pred_name)

        # Show the classes as the label on the x-axis.
        ax.set_xlabel(xlabel)

        # Remove ticks from the plot.
        ax.set_xticks([])
        ax.set_yticks([])

    # Ensure the plot is shown correctly with multiple plots.
    # in a single Notebook cell.
    plt.show()

# Plot a few images to see if data is correct
# Get the first images fro mthe test-set.
images = images_test[0:9]

# Get the true classes for those images.
cls_true = cls_test[0:9]

# Plot the images and labels using our helper-function above.
plot_images(images_=images, cls_true_=cls_true, smooth=False)
plot_images(images_=images, cls_true_=cls_true, smooth=True)

# TensorFlow Graph
# Placeholder variables
x = tf.placeholder(tf.float32, shape=[None, img_size, img_size, num_channels], name='x')
y_true = tf.placeholder(tf.float32, shape=[None, num_classes], name='y_true')
y_true_cls = tf.argmax(y_true, axis=1)

# Helper-function for creating Pre-Processing
def pre_process_image(image, training):
    # This function takes a single iamge as input,
    # and a boolean whether to build the training or testing graph.

    if training:
        # For training, add the following to the TensorFlow graph.

        # Randomly crop the input image.
        image = tf.random_crop(image, size=[img_size_cropped, img_size_cropped, num_channels])

        # Randomly flip the image horizontally.
        image = tf.image.random_flip_left_right(image)

        # Randomly adust hue, contrast and saturation.
        image = tf.image.random_hue(image, max_delta=0.05)
        image = tf.image.random_contrast(image, lower=0.3, upper=1.0)
        image = tf.image.random_brightness(image, max_delta=0.2)
        image = tf.image.random_saturation(image, lower=0.0, upper=2.0)

        # Some of these functions may overflow and result in pixel
        # values beyond the [0, 1] range. It is unclear from the
        # documentation of TensorFlow 0.10.0rc0 whether this is
        # intended. A simple solution is to limit the range.

        # Limit the image pixels between [0, 1] in case of overflow.
        image = tf.minimum(image, 1.0)
        image = tf.maximum(image, 0.0)
    else:
        # For test, add the following to the TensorFlow graph.

        # Crop the input image around the centre so it is the same
        # size as images taht are randomly cropped during training.
        image = tf.image.resize_image_with_crop_or_pad(image,
                                                       target_height=img_size_cropped,
                                                       target_width=img_size_cropped)

    return image

def pre_process(images_, training):
    # Use TensorFlow to loop over all the input images and call
    # the function above which takes a single image as input.
    images_ = tf.map_fn(lambda image: pre_process_image(image, training), images_)

    return images_

distorted_images = pre_process(images_=x, training=True)

# Helper-function for creating Main Processing
# Helper-function for creating new variables
def new_weights(shape, name):
    with tf.variable_scope(name):
        weights = tf.Variable(initial_value=tf.truncated_normal(shape, stddev=0.05), name='weights')

    return weights

def new_biases(length, name):
    with tf.variable_scope(name):
        biases = tf.Variable(initial_value=tf.constant(0.05, shape=[length]), name='biases')

    return biases

# Helper-function for creating a new Convolutional Layer
def new_conv_layer(input_,                          # The previous layer.
                   num_input_channels,              # Num. channels in prev. layer.
                   filter_size,                     # Width and height of each filter.
                   num_filters,                     # Number of filters.
                   use_pooling=True,
                   use_relu=True,
                   name=None):               # Use 2x2 max-pooling.
    # Shape of the filter-weights for the convolution.
    # This format is determined by the TensorFlow API.
    shape = [filter_size, filter_size, num_input_channels, num_filters]

    # Create new weights aka. filters with the given shape.
    weights = new_weights(shape=shape, name=name)

    # Create new biases, one for each filter.
    biases = new_biases(length=num_filters, name=name)

    # Create the TensorFlow operation for convolution.
    # Note the strides are set to 1 in all dimension.
    # The first and last stride must always be 1,
    # because the first is for the image-numer and
    # the last is for the input-channel.
    # But e.g. strides=[1, 2, 2, 1] would mean that the filter
    # is moved 2 pixels across the x- and y-axis of the image.
    # The padding is set to 'SAME' which means the input image
    # is padded with zeroses so teh size of the output is the same.
    layer = tf.nn.conv2d(input=input_,
                         filter=weights,
                         strides=[1, 1, 1, 1],
                         padding='SAME')

    # Add the biases to the resutls of the convolution.
    # A bias-value is added to each filter-channel.
    layer += biases

    # use pooling to down-sample the image resolution?
    if use_pooling:
        # This is 2x2 max-pooling, which mean taht we
        # consider 2x2 windows and select the largest value
        # in each window. Then we move 2 pixels to the next window.
        layer = tf.nn.max_pool(value=layer,
                               ksize=[1, 2, 2, 1],
                               strides=[1, 2, 2, 1],
                               padding='SAME')

    if use_relu:
        # Rectified Linear unit (ReLU).
        # It calculates max(x, 0) for each input pixel x.
        # This add some non-linearity to the formula and allows us
        # to learn more complicated functions.
        layer = tf.nn.relu(layer)

    # Note that ReLU is normally executed before the pooling,
    # but since relu(max_pool(x)) == max_pool(relu(x)) we can
    # save 75% of the relu-operations by max-pooling first.

    # We return both the resulting layer and the filter-weights
    # because we will plot the weights layer.
    return layer

# Helper-function for flattening a layer
def flatten_layer(layer):
    # Get the shape of the input layer.
    layer_shape = layer.get_shape()

    # The shape of the input layer is assumed to be:
    # layer_shape == [num_images, img_height, img_width, num_channels]

    # The number of features is: img_height * img_width * num_channels
    # We can use a function from TensorFlwo to calculate this.
    num_features = layer_shape[1:4].num_elements()

    # Reshape the layer to [num_images, num_features].
    # Note that we just set the size of the second dimension
    # to num_features and the size of the first dimension to -1
    # wihich means the size in that dimension is calculated
    # so the total size of the tensor is unchanged from the reshaping.
    layer_flat = tf.reshape(layer, [-1, num_features])

    # The shape of the flattened layer is now:
    # [num_images, img_height * img_width * num_channels]

    # Return both the flattened layer and the number of features.
    return layer_flat, num_features

# Helper-function for creating a new Fully-Connected Layer
def new_fc_layer(input_,                # The previous layer.
                 num_inputs,            # Num. inputs from prev. layer.
                 num_outputs,           # Num. outputs.
                 use_relu=True,         # Use Rectified Linear Unit (ReLU)?
                 name=None):

    # Create new weights and biases.
    weights = new_weights(shape=[num_inputs, num_outputs], name=name)
    biases = new_biases(length=num_outputs, name=name)

    # Calculate the layer as the matrix multiplication of
    # the input and weights, and then add the bias-values.
    layer = tf.matmul(input_, weights) + biases

    # Use ReLU?
    if use_relu:
        layer = tf.nn.relu(layer)

    return layer

def batch_norm(x, name, _ops, is_train=True):
    # Batch normalization
    with tf.variable_scope(name):
        params_shape = [x.get_shape()[-1]]

        beta = tf.get_variable(name='beta',
                               shape=params_shape,
                               dtype=tf.float32,
                               initializer=tf.constant_initializer(0., tf.float32))
        gamma = tf.get_variable(name='gamma',
                                shape=params_shape,
                                dtype=tf.float32,
                                initializer=tf.constant_initializer(1., tf.float32))

        if is_train:
            mean, variance = tf.nn.moments(x=x,
                                           axes=[0, 1, 2] if len(x.get_shape().as_list()) == 4 else [0],
                                           name='moments')

            moving_mean = tf.get_variable(name='moving_mean',
                                          shape=params_shape,
                                          dtype=tf.float32,
                                          initializer=tf.constant_initializer(value=0., dtype=tf.float32),
                                          trainable=False)
            moving_variance = tf.get_variable(name='moving_variance',
                                              shape=params_shape,
                                              dtype=tf.float32,
                                              initializer=tf.constant_initializer(value=1., dtype=tf.float32),
                                              trainable=False)

            _ops.append(moving_averages.assign_moving_average(variable=moving_mean,
                                                              value=mean,
                                                              decay=0.9))
            _ops.append(moving_averages.assign_moving_average(variable=moving_variance,
                                                              value=variance,
                                                              decay=0.9))
        else:
            mean = tf.get_variable(name='moving_mean',
                                   shape=params_shape,
                                   dtype=tf.float32,
                                   initializer=tf.constant_initializer(0., tf.float32),
                                   trainable=False)
            variance = tf.get_variable(name='moving_variance',
                                       shape=params_shape,
                                       dtype=tf.float32,
                                       initializer=tf.constant_initializer(1., tf.float32),
                                       trainable=False)

        # epsilon used to be 1e-5. Maybe 0.001 solves NaN problem in deeper net.
        y = tf.nn.batch_normalization(x=x,
                                      mean=mean,
                                      variance=variance,
                                      offset=beta,
                                      scale=gamma,
                                      variance_epsilon=1e-5)
        y.set_shape(x.get_shape())

        return y

def main_network(images, training):
    batch_ops = []

    layer_conv1 = new_conv_layer(input_=images,
                                 num_input_channels=num_channels,
                                 filter_size=filter_size1,
                                 num_filters=num_filters1,
                                 use_pooling=False,
                                 use_relu=False,
                                 name='layer_conv1')
    print('layer_conv1 shape: {}'.format(layer_conv1.get_shape().as_list()))

    layer_batch1 = batch_norm(x=layer_conv1, name='batch_1', _ops=batch_ops, is_train=training)
    print('layer_batch1 shape: {}'.format(layer_batch1.get_shape().as_list()))

    layer_relu1 = tf.nn.relu(features=layer_batch1)
    print('layer_relu1 shape: {}'.format(layer_relu1.get_shape().as_list()))

    layer_pool1 = tf.nn.max_pool(value=layer_relu1,
                                 ksize=[1, 2, 2, 1],
                                 strides=[1, 2, 2, 1],
                                 padding='SAME')
    print('layer_pool1 shape: {}'.format(layer_pool1.get_shape().as_list()))

    layer_conv2 = new_conv_layer(input_=layer_pool1,
                                 num_input_channels=num_filters1,
                                 filter_size=filter_size2,
                                 num_filters=num_filters2,
                                 use_pooling=False,
                                 use_relu=False,
                                 name='layer_conv2')
    print('layer_conv2 shape: {}'.format(layer_conv2.get_shape().as_list()))

    layer_batch2 = batch_norm(x=layer_conv2, name='batch_2', _ops=batch_ops, is_train=training)
    print('layer_batch2 shape: {}'.format(layer_batch2.get_shape().as_list()))

    layer_relu2 = tf.nn.relu(features=layer_batch2)
    print('layer_relu2 shape: {}'.format(layer_relu2.get_shape().as_list()))

    layer_pool2 = tf.nn.max_pool(value=layer_relu2,
                                 ksize=[1, 2, 2, 1],
                                 strides=[1, 2, 2, 1],
                                 padding='SAME')

    layer_flat, num_features = flatten_layer(layer_pool2)
    print('layer_flat shape: {}'.format(layer_flat.get_shape().as_list()))

    layer_fc1 = new_fc_layer(input_=layer_flat,
                             num_inputs=num_features,
                             num_outputs=fc_size1,
                             use_relu=False,
                             name='layer_fc1')
    print('layer_fc1 shape: {}'.format(layer_fc1.get_shape().as_list()))

    layer_batch3 = batch_norm(x=layer_fc1, name='batch_3', _ops=batch_ops, is_train=training)
    print('layer_batch3 shape: {}'.format(layer_batch3.get_shape().as_list()))

    layer_relu3 = tf.nn.relu(features=layer_batch3)
    print('layer_relu3 shape: {}'.format(layer_relu3.get_shape().as_list()))

    layer_fc2 = new_fc_layer(input_=layer_relu3,
                             num_inputs=fc_size1,
                             num_outputs=fc_size2,
                             use_relu=False,
                             name='layer_fc2')
    print('layer_fc2 shape: {}'.format(layer_fc2.get_shape().as_list()))

    layer_batch4 = batch_norm(x=layer_fc2, name='batch_4', _ops=batch_ops, is_train=training)
    print('layer_batch4 shape: {}'.format(layer_batch4.get_shape().as_list()))

    layer_relu4 = tf.nn.relu(features=layer_batch4)
    print('layer_relu4 shape: {}'.format(layer_relu4.get_shape().as_list()))

    y_pred = new_fc_layer(input_=layer_relu4,
                          num_inputs=fc_size2,
                          num_outputs=num_classes,
                          use_relu=False,
                          name='layer_fc3')
    print('y_pred shape: {}'.format(y_pred.get_shape().as_list()))

    loss = tf.nn.softmax_cross_entropy_with_logits_v2(logits=y_pred,
                                                      labels=y_true)

    return y_pred, loss, batch_ops

# Helper-function for creating Neural network
def create_network(training):
    # Wrap the neural network in the scope named 'network'.
    # Create new variables during training, and re-use during testing.
    with tf.variable_scope('network', reuse=not training):
        # Just rename the input placeholder varaible for convenience.
        images = x

        # Create TensorFlow graph for pre-processing.
        images = pre_process(images_=images, training=training)

        # Create TensorFlow graph for the main processing.
        y_pred, loss, batch_ops = main_network(images=images, training=training)

    return y_pred, loss, batch_ops

# Create Neural Network for Training Phase
global_step = tf.Variable(initial_value=0,
                          name='global_step',
                          trainable=False)

_, loss, batch_ops = create_network(training=True)

optimizer_ = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(loss=loss, global_step=global_step)
optimizer = [optimizer_] + batch_ops
optimizer = tf.group(*optimizer)

# Crate Neural Network for Test Phase / Inference
y_pred, _, _ = create_network(training=False)
y_pred_cls = tf.argmax(y_pred, axis=1)
correct_prediction = tf.equal(y_pred_cls, y_true_cls)
accuracy = tf.reduce_mean(input_tensor=tf.cast(x=correct_prediction, dtype=tf.float32))

# Saver
saver = tf.train.Saver()

# Getting the Weights
def get_weights_variable(layer_name):
    # Retrieve an existing variable named 'weights' in the scope
    # with the given layer_name.
    # This is awkward because the TensorFlow function was
    # really intended for another purpose.

    with tf.variable_scope(os.path.join('network', layer_name), reuse=True):
        variable = tf.get_variable('weights')

    return variable

weights_conv1 = get_weights_variable(layer_name='layer_conv1')
weights_conv2 = get_weights_variable(layer_name='layer_conv2')

# Getting the Layer Outputs
def get_layer_output(layer_name):
    # The name of the last operation of the convolutional layer.
    # This assumes you are using Relu as the activation-function
    tensor_name = os.path.join('network', layer_name, 'ReLu:0')

    # Get the tensor with this name.
    tensor = tf.get_default_graph().get_tensor_by_name(tensor_name)

    return tensor

output_conv1 = get_layer_output(layer_name='layer_conv1')
outptu_conv2 = get_layer_output(layer_name='layer_conv2')

# TensorFlow Run
# Crate TensorFlow session

# Restore or Initialize variables

# Helper-function to get a random training-batch
