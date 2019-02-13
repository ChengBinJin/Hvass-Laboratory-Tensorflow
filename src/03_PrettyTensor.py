import time
import math
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from datetime import timedelta

# We also need PrettyTensor.
import prettytensor as pt

from tensorflow.examples.tutorials.mnist import input_data

print('Tensorflow version: {}'.format(tf.__version__))
print('PrettyTensor version: {}'.format(pt.__version__))

# Load Data
data = input_data.read_data_sets('../data/MNIST/', one_hot=True)

print("Size of:")
print("- Training-set:\t\t{}".format(len(data.train.labels)))
print("- Test-set:\t\t{}".format(len(data.test.labels)))
print("- Validation-set:\t{}".format(len(data.validation.labels)))

data.test.cls = np.argmax(data.test.labels, axis=1)

# Data Dimensions
# We know that MNISt images are 28 pixels in each dimension.
img_size = 28

# images are stored in one-dimensional arrays of this length.
img_size_flat = img_size * img_size

# Tuple with height and width of images used to reshape arrays.
img_shape = (img_size, img_size)

# Number of colour channels for the images: 1 channel for gray-scale.
num_channels = 1

# Number of classes, one class for each 10 digits.
num_classes = 10

# Helper-function for plotting images
def plot_images(images_, cls_true_, cls_pred=None):
    assert len(images_) == len(cls_true_) == 9

    # Create figure with 3x3 sub-plots.
    fig, axes = plt.subplots(3, 3)
    fig.subplots_adjust(hspace=0.3, wspace=0.3)

    for i, ax in enumerate(axes.flat):
        # Plot image
        ax.imshow(images_[i].reshape(img_shape), cmap='binary')

        # Show true and predicted classes.
        if cls_pred is None:
            xlabel = "True: {0}".format(cls_true_[i])
        else:
            xlabel = "True: {0}, Pred: {1}".format(cls_true_[i], cls_pred[i])

        # Show the classes as the label on the x-axis.
        ax.set_xlabel(xlabel)

        # Remove ticks from the plot.
        ax.set_xticks([])
        ax.set_yticks([])

    # Ensure the plot is shown correctly with mutiple plots
    # in a single Notebook cell.
    plt.show()

# Get the first images from teh test-set
images = data.test.images[0:9]

# Get the true classes for those images.
cls_true = data.test.cls[0:9]

# Plot the images and labels using our helper-function above.
plot_images(images_=images, cls_true_=cls_true)

# Placeholder variables
x = tf.placeholder(tf.float32, shape=[None, img_size_flat], name='x')
x_image = tf.reshape(x, [-1, img_size, img_size, num_channels])
y_true = tf.placeholder(tf.float32, shape=[None, num_classes], name='y_true')
y_true_cls = tf.argmax(y_true, axis=1)

# TensorFlow Implementation
# Helper-functions
def new_weights(shape):
    return tf.Variable(tf.truncated_normal(shape, stddev=0.05))

def new_biases(length):
    return tf.Variable(tf.constant(0.05, shape=[length]))


def new_conv_layer(input_,              # The previous layer.
                   num_input_channels,  # Num. channels in prev. layer.
                   filter_size,         # Width and height of filters.
                   num_filters,         # Number of filters.
                   use_pooling=True):   # Use 2x2 max-pooling.

    # Shape of the filter-weights for the convolution.
    # Tis format is determined by the TensorFlow API.
    shape = [filter_size, filter_size, num_input_channels, num_filters]

    # Create new weights aka. filters with the given shape.
    weights = new_weights(shape=shape)

    # Create new biases, one for each filter.
    biases = new_biases(length=num_filters)

    # Create the TensorFlow operation for convolution.
    # Note the strides are set to 1 in all dimension.
    # The first and last stride must always be 1,
    # because the first is for the image-number and
    # the last is for the input-channel.
    # But e.g. strrides=[1, 2, 2, 1] would mean that the filter
    # is moved 2 pixels across the x- and y-axis of the image.
    # The padding is set to 'SAME' which means the input image
    # is padded with zeroes so the size of the outptu is the same.
    layer = tf.nn.conv2d(input=input_,
                         filter=weights,
                         strides=[1, 1, 1, 1],
                         padding='SAME')

    # Add the biases to the results of the convoltuion
    # A bias-value is added to each filter-channel.
    layer += biases

    # Use pooling to down-sample the image resolution?
    if use_pooling:
        # This is 2x2 max-pooling, which means that we
        # consider 2x2 windows and select the largest value
        # in each window. Then we move 2 pixels to the next window.
        layer = tf.nn.max_pool(value=layer,
                               ksize=[1, 2, 2, 1],
                               strides=[1, 2, 2, 1],
                               padding='SAME')

    # Rectified Linear Unit (ReLU).
    # It calculates max(x, 0) for each input pixel x.
    # This adds some non-linearity to the formula and allows us
    # to learn more complicated functions.
    layer = tf.nn.relu(layer)

    # Note that ReLU is normally executed before the pooling,
    # but since relu(max_pool(x)) == max_pool(relu(x)) we can
    # save 75% of the relu-operations by max-pooling first.

    # We return both the resulting layer and the filter-weights
    # because we will plot the weights later.
    return layer, weights

def flatten_layer(layer):
    # Get the shape of the input layer.
    layer_shape = layer.get_shape()

    # The shape of the input layer is assumed to be:
    # layer_shape == [num_images, img_height, img_width, num_channels]

    # The number of features is: img_height * img_width * num_channels
    # We can use a function from TensorFlow to calculate this.
    num_features_ = layer_shape[1:4].num_elements()

    # Reshape the layer to [num_iamges, num_features].
    # Note that we juset set the size of the second dimension
    # to num_featurs and the size of the first dimension to -1
    # which means the size in that dimension is calculated
    # so the total size of the tensor is unchaged from the reshaping.
    layer_flat_ = tf.reshape(layer, [-1, num_features_])

    # The shape of the flatten layer is now:
    # [num_images, img_height * img_width * num_channels]

    # Return both the flattened layer and the number of features.
    return layer_flat_, num_features_

def new_fc_layer(input_,            # The previous layer.
                 num_inputs,        # Num. inputs from prev. layer.
                 num_outputs,       # Num. outputs
                 use_relu=True):    # Use Rectified Linear Unit (ReLU)?

    # Create new weights and biases.
    weights = new_weights(shape=[num_inputs, num_outputs])
    biases = new_biases(length=num_outputs)

    # Calculate teh layer as the matrix multiplication of
    # the input and weights, and then add the bias-values.
    layer = tf.matmul(input_, weights) + biases

    # Use ReLU?
    if use_relu:
        layer = tf.nn.relu(layer)

    return layer

is_do = False
if is_do:   # Don't execute this! Just show it for easy comparsion.
    # First convolutional layer.
    layer_conv1, weights_conv1_ = new_conv_layer(input_=x_image,
                                                num_input_channels=num_channels,
                                                filter_size=5,
                                                num_filters=16,
                                                use_pooling=True)

    # Second convolutional layer.
    layer_conv2, weights_conv2_ = new_conv_layer(input_=layer_conv1,
                                                num_input_channels=16,
                                                filter_size=5,
                                                num_filters=36,
                                                use_pooling=True)

    # Flatten layer.
    layer_flat, num_features = flatten_layer(layer_conv2)

    # First fully-connected layer.
    layer_fc1 = new_fc_layer(input_=layer_flat,
                             num_inputs=num_features,
                             num_outputs=128,
                             use_relu=True)

    # Second fully-connected layer.
    layer_fc2 = new_fc_layer(input_=layer_fc1,
                             num_inputs=128,
                             num_outputs=num_classes,
                             use_relu=False)

    # Predicted class-label.
    y_pred_ = tf.nn.softmax(layer_fc2)

    # Cross-entropy for the classification of each image.
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(logits=layer_fc2,
                                                               labels=y_true)

    # Loss aka. cost-measure.
    # This is the scalar value that must be minimized.
    loss_ = tf.reduce_mean(cross_entropy)


# PrettyTensor Implementation
x_pretty = pt.wrap(x_image)

with pt.defaults_scope(activation_fn=tf.nn.relu):
    y_pred, loss = x_pretty.\
        conv2d(kernel=5, depth=16, name='layer_conv1').\
        max_pool(kernel=2, stride=2).\
        conv2d(kernel=5, depth=36, name='layer_conv2').\
        max_pool(kernel=2, stride=2).\
        flatten().\
        fully_connected(size=128, name='layer_fc1').\
        softmax_classifier(num_classes=num_classes, labels=y_true)

# Getting the weights
def get_weights_variable(layer_name):
    # Retrieve an existing variable named 'weights' in the scope
    # with the given layer_name.
    # This is awkward because teh TensorFlow function was
    # really intended for another purpose

    with tf.variable_scope(layer_name, reuse=True):
        variable = tf.get_variable('weights')

    return variable

weights_conv1 = get_weights_variable(layer_name='layer_conv1')
weights_conv2 = get_weights_variable(layer_name='layer_conv2')

# Optimization Method
optimizer = tf.train.AdamOptimizer(learning_rate=1e-4).minimze(loss)

# Performance Measure
y_pred_cls = tf.argmax(y_pred, dimension=1)
correct_prediction = tf.equal(y_pred_cls, y_true_cls)
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# TensorFlow Run
# Create TensorFlow session
session = tf.Session()

# Initialize variables
session.run(tf.global_variables_initializer())

# Helper-function to perform optimization iterations
train_batch_size = 64

# Counter for total number of iteratcions performed so far.
total_iterations = 0

def optimize(num_iterations):
    # Ensure we update the global variable rather than a local copy.
    global total_iterations

    # Start-time used for printing time-usage below.
    start_time = time.time()

    for i in range(total_iterations,
                   total_iterations + num_iterations):

        # Get a batch of training examples.
        # x_batch now holds a batch of images and
        # y_true_batch are the true labels for those images.
        x_batch, y_true_batch = data.train.next_batch(train_batch_size)

        # Put the batch into a dict with the proper names
        # for placeholder variables in the TensorFlow graph.
        feed_dict_train = {x: x_batch,
                           y_true: y_true_batch}

        # Run the optimizer using this batch of training data.
        # TensorFlow assign the variables in feed_dict_train
        # to the placeholder variables and then runs the optimizer.
        session.run(optimizer, feed_dict=feed_dict_train)

        # Print status every 100 iterations.
        if i % 100 == 0:
            # Calculate teh accuracy on the training-set.
            acc = session.run(accuracy, feed_dict=feed_dict_train)

            # Message for printing.
            msg = "Optimization Iteration: {0:>6}, Training Accuracy: {1:>6.1%}"

            # Print it
            print(msg.format(i + 1, acc))

    # Update the total number of iterations performed.
    total_iterations += num_iterations

    # Ending time
    end_time = time.time()

    # Difference between start and end-times.
    time_dif = end_time - start_time

    # Print the time-usage.
    print("Time usage: " + str(timedelta(seconds=int(round(time_dif)))))

# Helper-function to plot example errors
def plot_example_errors(cls_pred, correct):
    # This function is called from print_test_accuracy() below.

    # cls_pred is an array of the predicted class=nuber for
    # all images in the test-set.

    # correct is a boolean array whether the predicted class
    # is equal to the true class for each image in the test-set

    # negate the boolean array.
    incorrect = (correct == False)

    # Get the images from the test-set that have been
    # incorrectly classified.
    images_ = data.test.images[incorrect]

    # Get the predicted classes for those images.
    cls_pred = cls_pred[incorrect]

    # Get the true classes for those images.
    cls_true_ = data.test.cls[incorrect]

    # Plot the first 9 images
    plot_images(images_=images_[0:9],
                cls_true_=cls_true_[0:9],
                cls_pred=cls_pred[0:9])

# Helper-function to plot confusion matrix
def plot_confusion_matrix(cls_pred):
    # This is called from print_test_accuracy() below.

    # cls_pred is an array of the predicted class-number for
    # all images in the test-set

    # Get the true classifications for the test-set.
    cls_true_ = data.test.cls

    # Get the confusion matrix using sklearn.
    cm = confusion_matrix(y_true=cls_true_,
                          y_pred=cls_pred)

    # Print the confusion matrix as text.
    print(cm)

    # Plot the confusion matrix as an image.
    plt.matshow(cm)

    # Make various adjustments to the plot.
    plt.colorbar()
    tick_marks = np.arange(num_classes)
    plt.xticks(tick_marks, range(num_classes))
    plt.yticks(tick_marks, range(num_classes))
    plt.xlabel('Predicted')
    plt.ylabel('True')

    # Ensure the plot is shown correctly with multiple plots
    # in a single Notebook cell.
    plt.show()

# Helper-function for showing the performance
# Split the test-set into smaller batches of this size.
test_batch_size = 256

def print_test_accuracy(show_example_errors=False,
                        show_confusion_matrix=False):
    # Number of images in the test-set.
    num_test = len(data.test.images)

    # Allocate an array for the predicted classes which
    # will be calculated in batches and filled into this array.
    cls_pred = np.zeros(shape=num_test, dtype=np.int)

    # Now calculate the predicted classes for the batches.
    # We will just iterate through all the batches.
    # There might be a more clever and Pythonic way of doing this.

    # The starting index of the next batch is denoted i.
    i = 0

    while i < num_test:
        # The ending index for the next batch is denoted j.
        j = min(i + test_batch_size, num_test)

        # Get the iamges from the test-set between index i and j.
        images_ = data.test.images[i:j, :]

        # Get the associated labels.
        labels =data.test.labels[i:j, :]

        # Create a feed-dict with these images and labels.
        feed_dict = {x: images_,
                     y_true: labels}

        # Calculate the predicted class using TensorFlow.
        cls_pred[i:j] = session.run(y_pred_cls, feed_dict=feed_dict)

        # Set the start-index for the next batch to the
        # end-index of the current batch.
        i = j

    # Convenience variable for the true class-numbers of the test-set.
    cls_true_ = data.test.cls

    # Create a boolean array wheter each image is orrectly classified.
    correct = (cls_true_ == cls_pred)

    # Calculate the number of correctly classified images.
    # When summing a boolean array, False mean 0 and True means 1.
    correct_sum = correct.sum()

    # Classification accuracy is the number of correctly classified
    # images divided by the total number of images in the test-set.
    acc = float(correct_sum) / num_test

    # Print the accuracy.
    msg = "Accuracy on Test-set: {0:.1%} ({1} / {2})"
    print(msg.format(acc, correct_sum, num_test))

    # Plot some examples off mis-classifications, if desired.
    if show_example_errors:
        print("Example errors:")
        plot_example_errors(cls_pred=cls_pred, correct=correct)

    # Plot the confusion matrix, if desired.
    if show_confusion_matrix:
        print("Confusion Matrix:")
        plot_confusion_matrix(cls_pred=cls_pred)

# Performance before any optimization
print_test_accuracy()

# Performance after 1 optimization iteration
optimize(num_iterations=1)
print_test_accuracy()

# Performance after 100 optimization iterations
optimize(num_iterations=99)  # We already performed 1 iteration above.
print_test_accuracy()

# Performance after 1000 optimization iterations
optimize(num_iterations=900)  # We performed 100 iterations above.
print_test_accuracy(show_example_errors=True)

# Performance after 10,000 optimization iteations
optimize(num_iterations=9000)  # We performed 1000 iterations above.
print_test_accuracy(show_example_errors=True,
                    show_confusion_matrix=True)

# Visualization of Weights and Layers
# Helper-function for plotting convolutional weights
def plot_conv_weights(weights, input_channel=0):
    # Assume weights are TensorFlow ops for 4-dim variables
    # e.g. weights_conv1 or weights_conv2.

    # Retrieve the values of the weight-variables from TensorFLow.
    # A feed-dict is not necessary because nothing is calculated.
    w = session.run(weights)

    # Get the lowest and highest values for the weights.
    # This is used to correct the colour intensity across
    # the images so they can be compared with each other.
    w_min = np.min(w)
    w_max = np.max(w)

    # Number of filters used in the conv. layer.
    num_filters = w.shape[3]

    # Number of grids to plot.
    # Rounded-up, square-root of the number of filters.
    num_grids = math.ceil(math.sqrt(num_filters))

    # Create figure with a grid of sub-plots.
    fig, axes = plt.subplots(num_grids, num_grids)

    # Plot all the filter-weights.
    for i, ax in enumerate(axes.flat):
        # Only plot the valid filter-weights.
        if i < num_filters:
            # Get the weights for the i'th filter of the input channel.
            # See new_conv_layer() for details on the format
            # of this 4-dim tensor.
            img = w[:, :, input_channel, i]

            # Plot image.
            ax.imshow(img, vmin=w_min, vmax=w_max, interpolation='nearest', cmap='seismic')

        # Remove ticks from the plot.
        ax.set_xticks([])
        ax.set_yticks([])

    # Ensure the plot is shown correctly with multiple plots
    # in a single Notebook cell.
    plt.show()


# Convolutional Layer 1
plot_conv_weights(weights=weights_conv1)

# Convolutaionl Layer 2
plot_conv_weights(weights=weights_conv2, input_channel=0)
plot_conv_weights(weights=weights_conv2, input_channel=1)



