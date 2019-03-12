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
def new_weights(shape):
    weights = tf.get_variable(initializer=tf.truncated_normal(shape, stddev=0.05), name='weights')
    return weights

def new_biases(length):
    biases = tf.get_variable(initializer=tf.constant(0.05, shape=[length]), name='biases')
    return biases

# Helper-function for creating a new Convolutional Layer
def new_conv_layer(input_,                          # The previous layer.
                   num_input_channels,              # Num. channels in prev. layer.
                   filter_size,                     # Width and height of each filter.
                   num_filters,                     # Number of filters.
                   use_pooling=True,
                   use_relu=True):                  # Use 2x2 max-pooling.
    # Shape of the filter-weights for the convolution.
    # This format is determined by the TensorFlow API.
    shape = [filter_size, filter_size, num_input_channels, num_filters]

    # Create new weights aka. filters with the given shape.
    weights = new_weights(shape=shape)

    # Create new biases, one for each filter.
    biases = new_biases(length=num_filters)

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
                 use_relu=True):        # Use Rectified Linear Unit (ReLU)?

    # Create new weights and biases.
    weights = new_weights(shape=[num_inputs, num_outputs])
    biases = new_biases(length=num_outputs)

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

    with tf.variable_scope('layer_conv1'):
        layer_conv1 = new_conv_layer(input_=images,
                                     num_input_channels=num_channels,
                                     filter_size=filter_size1,
                                     num_filters=num_filters1,
                                     use_pooling=False,
                                     use_relu=False)
        print('layer_conv1 shape: {}'.format(layer_conv1.get_shape().as_list()))

        layer_batch1 = batch_norm(x=layer_conv1, name='batch_1', _ops=batch_ops, is_train=training)
        print('layer_batch1 shape: {}'.format(layer_batch1.get_shape().as_list()))

        layer_relu1 = tf.nn.relu(features=layer_batch1)
        print('layer_relu1 shape: {}'.format(layer_relu1.get_shape().as_list()))
        print(layer_relu1)

        layer_pool1 = tf.nn.max_pool(value=layer_relu1,
                                     ksize=[1, 2, 2, 1],
                                     strides=[1, 2, 2, 1],
                                     padding='SAME')
        print('layer_pool1 shape: {}'.format(layer_pool1.get_shape().as_list()))

    with tf.variable_scope('layer_conv2'):
        layer_conv2 = new_conv_layer(input_=layer_pool1,
                                     num_input_channels=num_filters1,
                                     filter_size=filter_size2,
                                     num_filters=num_filters2,
                                     use_pooling=False,
                                     use_relu=False)
        print('layer_conv2 shape: {}'.format(layer_conv2.get_shape().as_list()))

        layer_batch2 = batch_norm(x=layer_conv2, name='batch_2', _ops=batch_ops, is_train=training)
        print('layer_batch2 shape: {}'.format(layer_batch2.get_shape().as_list()))

        layer_relu2 = tf.nn.relu(features=layer_batch2)
        print('layer_relu2 shape: {}'.format(layer_relu2.get_shape().as_list()))
        print(layer_relu2)

        layer_pool2 = tf.nn.max_pool(value=layer_relu2,
                                     ksize=[1, 2, 2, 1],
                                     strides=[1, 2, 2, 1],
                                 padding='SAME')

    layer_flat, num_features = flatten_layer(layer_pool2)
    print('layer_flat shape: {}'.format(layer_flat.get_shape().as_list()))

    with tf.variable_scope('layer_fc1'):
        layer_fc1 = new_fc_layer(input_=layer_flat,
                                 num_inputs=num_features,
                                 num_outputs=fc_size1,
                                 use_relu=False)
        print('layer_fc1 shape: {}'.format(layer_fc1.get_shape().as_list()))

        layer_batch3 = batch_norm(x=layer_fc1, name='batch_3', _ops=batch_ops, is_train=training)
        print('layer_batch3 shape: {}'.format(layer_batch3.get_shape().as_list()))

        layer_relu3 = tf.nn.relu(features=layer_batch3)
        print('layer_relu3 shape: {}'.format(layer_relu3.get_shape().as_list()))

    with tf.variable_scope('layer_fc2'):
        layer_fc2 = new_fc_layer(input_=layer_relu3,
                                 num_inputs=fc_size1,
                                 num_outputs=fc_size2,
                                 use_relu=False)
        print('layer_fc2 shape: {}'.format(layer_fc2.get_shape().as_list()))

        layer_batch4 = batch_norm(x=layer_fc2, name='batch_4', _ops=batch_ops, is_train=training)
        print('layer_batch4 shape: {}'.format(layer_batch4.get_shape().as_list()))

        layer_relu4 = tf.nn.relu(features=layer_batch4)
        print('layer_relu4 shape: {}'.format(layer_relu4.get_shape().as_list()))

    with tf.variable_scope('layer_fc3'):
        y_pred = new_fc_layer(input_=layer_relu4,
                              num_inputs=fc_size2,
                              num_outputs=num_classes,
                              use_relu=False)
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
# global_step = tf.Variable(initial_value=0,
#                           name='global_step',
#                           trainable=False)
global_step = tf.get_variable(initializer=0,
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

def show_all_variables():
    total_count = 0
    for idx, op in enumerate(tf.trainable_variables()):
        shape = op.get_shape()
        count = np.prod(shape)
        print("[%2d] %s %s = %s" % (idx, op.name, shape, count))
        total_count += int(count)
    print("[Total] variable size: %s" % "{:,}".format(total_count))

show_all_variables()

# Getting the Weights
def get_weights_variable(layer_name):
    # Retrieve an existing variable named 'weights' in the scope
    # with the given layer_name.
    # This is awkward because the TensorFlow function was
    # really intended for another purpose.

    with tf.variable_scope('network/' + layer_name, reuse=True):
        variable = tf.get_variable(name='weights')

    return variable

weights_conv1 = get_weights_variable(layer_name='layer_conv1')
weights_conv2 = get_weights_variable(layer_name='layer_conv2')

# Getting the Layer Outputs
def get_layer_output(layer_name):
    # The name of the last operation of the convolutional layer.
    # This assumes you are using Relu as the activation-function
    tensor_name = 'network/' + layer_name + '/Relu:0'

    # Get the tensor with this name.
    tensor = tf.get_default_graph().get_tensor_by_name(tensor_name)

    return tensor

output_conv1 = get_layer_output(layer_name='layer_conv1')
outptu_conv2 = get_layer_output(layer_name='layer_conv2')

# TensorFlow Run
# Creat TensorFlow session
session = tf.Session()

# Restore or Initialize variables
save_dir = '../checkpoints/'

if not os.path.exists(save_dir):
    os.makedirs(save_dir)

save_path = os.path.join(save_dir, 'cifar10_cnn')

try:
    print("Trying to restore last checkpoint ...")

    # Use TensorFlow to find the latest checkpoint - if any.
    last_chk_path = tf.train.latest_checkpoint(checkpoint_dir=save_dir)

    # Try and load the data in the checkpoint.
    saver.restore(session, save_path=last_chk_path)

    # If we get to this point, the checkpoint was successfully loaded.
    print("Restored checkpoint from:", last_chk_path)
except:
    # If the above failed for some reason, simply
    # initialize all the variables for the TensorFlow graph.
    print("Failed to restore checkpoint. Initializing variables instead.")
    session.run(tf.global_variables_initializer())

# Helper-function to get a random training-batch
train_batch_size = 64

def random_batch():
    # Number of images in the training-set.
    num_images = len(images_train)

    # Create a random index.
    idx = np.random.choice(num_images,
                           size=train_batch_size,
                           replace=False)

    # Use the random index to select random images and labels.
    x_batch = images_train[idx, :, :, :]
    y_batch = labels_train[idx, :]

    return x_batch, y_batch

# Helper-function to perform optimization
def optimize(num_iterations):
    # Start-time used for printing time-usage below.
    start_time = time.time()

    for i in range(num_iterations):
        # Get a batch of training examples.
        # x_batch now holds a batch of images and
        # y_true_batch are the true labels for those images.
        x_batch, y_true_batch = random_batch()

        # Put the batch into a dict with the proper names
        # for placeholder variables in the TensorFlow graph.
        feed_dict_train = {x: x_batch,
                           y_true: y_true_batch}

        # Run the optimizer using this batch of training data.
        # TensorFlow assigns the variables in feed_dict_train
        # to the placeholder variables and then runs the optimizer.
        # We also want to retrieve the global_step counter.
        i_global, _ = session.run([global_step, optimizer],
                                  feed_dict=feed_dict_train)

        # Print status to screen every 100 iterations (and last).
        if (i_global % 100 == 0) or (i == num_iterations - 1):
            # Calculate the accuracy on the training-batch.
            batch_acc = session.run(accuracy,
                                    feed_dict=feed_dict_train)

            # Print status.
            msg = "Global Step: {0:>6}, Training Batch Accuracy: {1:>6.1%}"
            print(msg.format(i_global, batch_acc))

        # Save a checkpoint to disk every 1000 iterations (and last).
        if (i_global % 1000 == 0) or (i == num_iterations - 1):
            # Save all variables of the TensorFlow graph to a
            # checkpoint. Append the global_step counter
            # to the filename so we save the last several checkpoints.
            saver.save(session,
                       save_path=save_path,
                       global_step=global_step)

            print("Saved checkpoint.")

    # Ending time.
    end_time = time.time()

    # Difference between start and end-times.
    time_dif = end_time - start_time

    # Print the time-usage.
    print("Time usage: " + str(timedelta(seconds=int(round(time_dif)))))

# Helper-function to plot example errors
def plot_example_errors(cls_pred, correct):
    # This function is called from print_test_accuracy() below.

    # cls_pred is an array of the predicted class-number for
    # all images in the test-set.

    # correct is a boolean array whether the predicted class
    # is equal to the true class for each image in the test-set.

    # Negate the boolean array.
    incorrect = (correct == False)

    # Get the images from the test-set that have been
    # incorrectly classified.
    images = images_test[incorrect]

    # Get the predicted classes for those images.
    cls_pred = cls_pred[incorrect]

    # Get the true classes for those images.
    cls_true = cls_test[incorrect]

    # Plot the first 9 images.
    plot_images(images_=images[0:9],
                cls_true_=cls_true[0:9],
                cls_pred=cls_pred[0:9])

# Helper-function to plot confusion matrix
def plot_confusion_matrix(cls_pred):
    # This is called from print_test_accuracy() below.

    # cls_pred is an array of the predicted class-number for
    # all images in the test-set.

    # Get the confusion matrix using sklearn.
    cm = confusion_matrix(y_true=cls_test,      # True class for test-set.
                          y_pred=cls_pred)      # Predicted class.

    # Print the confusion matrix as text.
    for i in range(num_classes):
        # Append the class-name to each line.
        class_name = "({}) {}".format(i, class_names[i])
        print(cm[i, :], class_name)

    # Print the class-numbers for easy reference.
    class_numbers = [" ({0})".format(i) for i in range(num_classes)]
    print("".join(class_numbers))

# Helper-functions for calculating classifications
# Split the data-set in batches of this size to limit RAM usage.
batch_size = 256

def predict_cls(images, labels, cls_true):
    # Number of images.
    num_images = len(images)

    # Allocate an array for the predicted classes which
    # will be calculated in batches and filled into this array.
    cls_pred = np.zeros(shape=num_images, dtype=np.int)

    # Now calculate the predicted classes for the batches.
    # We will just iterate through all the batches.
    # There might be a more clever and Pythonic way of doing this.

    # This starting index for the next batch is denoted i.
    i = 0

    while i < num_images:
        # The ending index for the next batch is denoted j.
        j = min(i + batch_size, num_images)

        # Create a feed-dict with the images and labels
        # between index i and j.
        feed_dict = {x: images[i:j, :],
                     y_true: labels[i:j, :]}

        # Calculate the predicted class using TensorFlow.
        cls_pred[i:j] = session.run(y_pred_cls, feed_dict=feed_dict)

        # Set the start-index for the next batch to the
        # end-index of the current batch.
        i = j

    # Create a boolean array whether each image is correctly classified.
    correct = (cls_true == cls_pred)

    return correct, cls_pred

def predict_cls_test():
    return predict_cls(images = images_test,
                       labels = labels_test,
                       cls_true = cls_test)

# Helper-functions for the classification accuracy
def classification_accuracy(correct):
    # When averaging a boolean array, False means 0 and True means 1.
    # So we are calculating: number of True / len(correct) which is
    # the same as the classification accuracy.

    # Return the classification accuracy
    # and the number of correct classifications.
    return correct.mean(), correct.sum()

# Helper-function for showing the performance
def print_test_accuracy(show_example_errors=False,
                        show_confusion_matrix=False):
    # For all the images in the test-set,
    # calculate the predicted classes and whether they are correct.
    correct, cls_pred = predict_cls_test()

    # Classification accuracy and the number of correct classifications.
    acc, num_correct = classification_accuracy(correct)

    # Number of images being classified.
    num_images = len(correct)

    # Print the accuracy.
    msg = "Accuracy on Test-Set: {0:.1%} ({1} / {2})"
    print(msg.format(acc, num_correct, num_images))

    # Plot some examples of mis-classifications, if desired.
    if show_example_errors:
        print("Example errors:")
        plot_example_errors(cls_pred=cls_pred, correct=correct)

    # Plot the confusion matrix, if desired.
    if show_confusion_matrix:
        print("Confusion matrix:")
        plot_confusion_matrix(cls_pred=cls_pred)

# Helper-function for plotting convolutional weights
def plot_conv_weights(weights, input_channel=0):
    # Assuem weights are TensorFlow ops for 4-dim varaibles
    # e.g. weights_conv1 or weights_conv2

    # Retrievethe values of the weight-variables from TensorFlow.
    # A feed-dict is not necessary because nothing is calculated.
    w = session.run(weights)

    # Print statistics for the weights.
    print("Min:     {0:.5f}, Max:   {1:.5f}".format(w.min(), w.max()))
    print("Mean:    {0:.5f}, Stdev: {1:.5f}".format(w.mean(), w.std()))

    # Get the lowest and highest values for the weights.
    # This is used to correct the colour intensity across
    # the images so they can be compared with each other.
    w_min = np.min(w)
    w_max = np.max(w)
    abs_max = max(abs(w_min), abs(w_max))

    # Number of filters used in the conv. layer.
    num_filters = w.shape[3]

    # Number of grids to plot.
    # Rounded-up, square-root of the number of filters.
    num_grids = math.ceil(math.sqrt(num_filters))

    # Create figure with  grid of sub-plots.
    fig, axes = plt.subplots(num_grids, num_grids)

    # Plot all the filter-weights.
    for i, ax in enumerate(axes.flat):
        # Only plot the valid filter-weights.
        if i < num_filters:
            # Get the weights for the i't h filter of the input channel.
            # Teh format of this 4-dim tensor is determined by the
            # TensorFlow API. See Tutorial #02 for more details.
            img = w[:, :, input_channel, i]

            # Plot image.
            ax.imshow(img, vmin=-abs_max, vmax=abs_max,
                      interpolation='nearest', cmap='seismic')

        # Remove ticks from the plot.
        ax.set_xticks([])
        ax.set_yticks([])

    # Ensure the plot is shown correctly with multiple plots
    # in a single Notebook cell.
    plt.show()

# Helper-function for plotting the output of convolutional layers
def plot_layer_output(layer_output, image):
    # Assume layer_output is a 4-dim tensor
    # e.g. output_conv1 or output_conv2.

    # Create a feed-dict which holds the single input image.
    # Note taht TensorFlow needs a list of images.
    # so we just create a list with this one image.
    feed_dict = {x: [image]}

    # Retrieve the output of the layer after inputting this iamge.
    values = session.run(layer_output, feed_dict=feed_dict)

    # Get the lowest and highest values.
    # This is used to correct the colour intensity across
    # the iamges so they can be compared with each oterh.
    values_min = np.min(values)
    values_max = np.max(values)

    # Number of image channels output by the conv. layer.
    num_images = values.shape[3]

    # Number of grid-cells to plot.
    # Rounded-up, square-root of the number of filters.
    num_grids = math.ceil(math.sqrt(num_images))

    # Create figure with a grid of sub-plots.
    fig, axes = plt.subplots(num_grids, num_grids)

    # Plot all the filter-weights.
    for i, ax in enumerate(axes.flat):
        # Only plot the valid image-channels.
        if i < num_images:
            # Get the images for the i'th output channel.
            img = values[0, :, :, i]

            # Plot iamge.
            ax.imshow(img, vmin=values_min, vmax=values_max,
                      interpolation='nearest', cmap='binary')

        # Remove ticks from the plot.
        ax.set_xticks([])
        ax.set_yticks([])

    # Ensure the plot is shown correctly with multiple plots
    # in a single Notebook cell.
    plt.show()

# Examples of distorted input images
def plot_distorted_image(image, cls_true):
    # Repeat the input image 9 times.
    image_duplicates = np.repeat(image[np.newaxis, :, :, :], 9, axis=0)

    # Create a feed-dict for TensorFlow.
    feed_dict = {x: image_duplicates}

    # Calculate only the pre-processing of the tensorFlow graph
    # which distorts the images in the feed-dict.
    result = session.run(distorted_images, feed_dict=feed_dict)

    # Plot the images.
    plot_images(images_=result, cls_true_=np.repeat(cls_true, 9))

def get_test_image(i):
    return images_test[i, :, :, :], cls_test[i]

img, cls = get_test_image(16)

plot_distorted_image(image=img, cls_true=cls)

# Perform optimization
is_run = True
if is_run:
    optimize(num_iterations=150000)  # 150,000

# Results
print_test_accuracy(show_example_errors=True,
                    show_confusion_matrix=True)

# Convolutional Weights
plot_conv_weights(weights=weights_conv1, input_channel=0)
plot_conv_weights(weights=weights_conv2, input_channel=0)

# Output of convolutional layers
def plot_image(image):
    # Create figure with sub-plots.
    fig, axes = plt.subplots(1, 2)

    # References to the sub-plots.
    ax0 = axes.flat[0]
    ax1 = axes.flat[1]

    # Show raw and smoothened images in sub-plots.
    ax0.imshow(image, interpolation='nearest')
    ax1.imshow(image, interpolation='spline16')

    # Set labels
    ax0.set_xlabel('Raw')
    ax1.set_xlabel('Smooth')

    # Ensure the plot is shown correctly with multiple plots
    # in a single Notebook cell.
    plt.show()

img, cls = get_test_image(16)
plot_image(img)

plot_layer_output(output_conv1, image=img)
plot_layer_output(outptu_conv2, image=img)

# Predicted class-labels
label_pred, cls_pred = session.run([y_pred, y_pred_cls],
                                   feed_dict={x: [img]})

# Set the ronding options for numpy.
np.set_printoptions(precision=3, suppress=True)

# Print the predicted label.
print(label_pred[0])

for idx in range(len(class_names)):
    print(idx, class_names[idx])

# Close TensorFlow Session
# This has been commented out in case you want to modify and experiment
# with the Notebook without having to restart it.
session.close()
