import os
import math
import time
import math
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.metrics import confusion_matrix
from datetime import timedelta

from mnist import MNIST

print("Tensorflow version: {0}".format(tf.__version__))

# Configuration of Neural Network
# Convolutional Layer 1.
filter_size1 = 5        # Convolution filters are 5 x 5 pixels.
num_filters1 = 16       # There are 16 of these filters.

# Convolutional Layer 2.
filter_size2 = 5        # Convolution filters are 5 x 5 pixels.
num_filters2 = 36       # ther are 36 of these filters.

# Fully-connected layer.
fc_size = 128           # Number of neurons in fully-connected layer.

# Load data
data = MNIST(data_dir="../data/MNIST/")

print("Size of:")
print("- Training-set:\t\t{}".format(data.num_train))
print("- Validation-set:\t{}".format(data.num_val))
print("- Test-set:\t\t{}".format(data.num_test))

# The number of pixels in each dimension of an image.
img_size = data.img_size

# The images are stored in one-dimensional arrays of this length.
img_size_flat = data.img_size_flat

# Tuple with height and widht of images used to reshape arrays.
img_shape = data.img_shape

# Number of classes, one class for each of 10 digits.
num_classes = data.num_classes

# Number of colour channels for the images: 1 channel for gray-scale.
num_channels = data.num_channels

# Helper-function for plotting images
def plot_images(images_, cls_true_, cls_pred=None):
    assert len(images_) == len(cls_true) == 9

    # Create figure with 3x3 xub-plots.
    fig, axes = plt.subplots(3, 3)
    fig.subplots_adjust(hspace=0.3, wspace=0.3)

    for i, ax in enumerate(axes.flat):
        # Plot iamge.
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

    # Ensure the plot is shown correctly with multiple plots
    # in a single Notebook cell.
    plt.show()

# Plot a few images to see if data is correct
# Get the first images from the test-set.
images = data.x_test[0:9]

# Get the true classes for those images.
cls_true = data.y_test_cls[0:9]

# Plot the images and labels using our helper-function above.
plot_images(images_=images, cls_true_=cls_true)

# TensorFlwo Graph
# Placeholder variables
x = tf.placeholder(tf.float32, shape=[None, img_size_flat], name='x')
x_iamge = tf.reshape(x, [-1, img_size, img_size, num_channels])
y_true = tf.placeholder(tf.float32, [None, 10], name='y_ture')
y_true_cls = tf.argmax(y_true, dimension=1)

# Helper-functions for creating new variables
def new_weights(shape):
    return tf.Variable(tf.truncated_normal(shape, stddev=0.05))

def new_biases(length):
    return tf.Variable(tf.constant(0.05, shape=[length]))

# Helper-function for creating a new Convolutional Layer
def new_conv_layer(input_,              # The previous layer.
                   num_input_channels,  # Num. channels in prev. layer.
                   filter_size,         # Width and heigth of each filter.
                   num_filters,         # Number of filters.
                   use_pooling=True):   # Use 2x2 max-pooling.

    # Shape of the filter-weights for the convolution.
    # This format is determined by the TensorFlow API.
    shape = [filter_size, filter_size, num_input_channels, num_filters]

    # Crate new weights aka. filters with the given shape.
    weights = new_weights(shape=shape)

    # Create new biases, one for each filter.
    biases = new_biases(length=num_filters)

    # Create the TensorFlow operation for convolution.
    # Note the strides are set to 1 in all dimension.
    # The first and last stride must always be 1,
    # because the first is for the image-number and
    # the last is for the input-channel.
    # But e.g. strides=[1, 2, 2, 1] would mean that the filter
    # is moved 2 pixels across the x- and y-axis of the image.
    # The padding is set to 'SAME' which means the input image
    # is padded with zeros so the size of the output is the same.
    layer = tf.nn.conv2d(input=input_,
                         filter=weights,
                         strides=[1, 1, 1, 1],
                         padding='SAME')

    # Add the biases to the results of the convolution.
    # A bias-value is added to eah filter-channel.
    layer += biases

    # Use pooling to down-sample the image resolution?
    if use_pooling:
        # This is 2x2 max-pooling, which mean that we
        # consider 2x2 windows and select the largest value
        # in each window. Then we move 2 pixels to the next window.
        layer = tf.nn.max_pool(value=layer,
                               ksize=[1, 2, 2, 1],
                               strides=[1, 2, 2, 1],
                               padding='SAME')

    # Rectified Linear Unit (ReLU).
    # It acalcuates max(x, 0) for each input pixel x.
    # This adds some non-linearity to the formula and allows us
    # to learn more complicated functions.
    layer = tf.nn.relu(layer)

    # Note that ReLU is normally executed before the pooling,
    # but since relu(max_pool(x)) == max_pool(relu(x)) we can
    # save 75% of the relu-operations by max-pooling first.

    # We return both the resulting layer and the filter-weights
    # because we will plot the weights layer.
    return layer, weights

# Helper-function for flattening a layer
def flatten_layer(layer):
    # Get the shape of the input layer.
    layer_shape = layer.get_shape()

    # The shape of the input layer is assumed to be:
    # layer_shape == [num_images, im_height, img_width, num_channels]

    # The number of features is: img_height * img_width * num_channels
    # We can use a function from TensorFlow to calculate this.
    num_features_ = layer_shape[1:4].num_elements()

    # Reshape the layer to [num_images, num_features].
    # Note taht we just set the size of the second dimension
    # to num_featurs and teh size of the first dimension to -1
    # which means the sizes in that dimension is calculated
    # so the total isze of the tensor is unchanged from the reshaping.
    layer_flat_ = tf.reshape(layer, [-1, num_features_])

    # The shape of the flattened layer is now:
    # [num_images, img_height * img_width * num_channels]

    # Return bot hthe flattened layer and the number of features.
    return layer_flat_, num_features_

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

# Convolutional Layer 1
layer_conv1, weight_conv1 = new_conv_layer(input_=x_iamge,
                                           num_input_channels=num_channels,
                                           filter_size=filter_size1,
                                           num_filters=num_filters1,
                                           use_pooling=True)
print("layer_conv1 shape: {}".format(layer_conv1.get_shape().as_list()))

# Convolutional Layer 2
layer_conv2, weights_conv2 = new_conv_layer(input_=layer_conv1,
                                            num_input_channels=num_filters1,
                                            filter_size=filter_size2,
                                            num_filters=num_filters2,
                                            use_pooling=True)
print("layer_conv2 shape: {}".format(layer_conv2.get_shape().as_list()))

# Flatten Layer
layer_flat, num_features = flatten_layer(layer_conv2)
print("layer_flat shape: {}".format(layer_flat.get_shape().as_list()))
print("num_feaures: {}".format(num_features))

# Fully-Connected Layer 1
layer_fc1 = new_fc_layer(input_=layer_flat,
                         num_inputs=num_features,
                         num_outputs=fc_size,
                         use_relu=True)
print("layer_fc1 shape: {}".format(layer_fc1.get_shape().as_list()))

# Fully-Connected Layer 2
layer_fc2 = new_fc_layer(input_=layer_fc1,
                         num_inputs=fc_size,
                         num_outputs=num_classes,
                         use_relu=False)
print("layer_fc2 shape: {}".format(layer_fc2.get_shape().as_list()))

# Predicted Class
y_pred = tf.nn.softmax(layer_fc2)
y_pred_cls = tf.argmax(y_pred, axis=1)

# Cost-function to be optimized
cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(logits=layer_fc2, labels=y_true)
cost = tf.reduce_mean(cross_entropy)

# Optimization Method
optimizer = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(cost)

# Performance Measures
correct_prediction = tf.equal(y_pred_cls, y_true_cls)
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# Saver
saver = tf.train.Saver()
save_dir = '../checkpoints/'

if not os.path.exists(save_dir):
    os.makedirs(save_dir)

save_path = os.path.join(save_dir, 'best_validation')

# TensorFlow Run
# Create TensorFlow session
session = tf.Session()

def init_variables():
    session.run(tf.global_variables_initializer())

init_variables()

# Helper-function to perform optimization iterations
train_batch_size = 64

# Best validatio naccuracy seen so far.
best_validation_accuracy = 0.0

# Iteration-number for last improvement to validation accuracy.
last_improvement = 0

# Stop optimization if no improvment found in this many iterations.
require_improvement = 1000

# Counter for total number of iterations performed so far.
total_iterations = 0

# Helper-functions for the classification accuracy
def cls_accuracy(correct):
    # Calculate the number of correctly classified images.
    # Whe nsumming a boolean array, False means 0 and True means 1.
    correct_sum = correct.sum()

    # Classification accuracy is the number of correctly classified
    # images divided by the total number of images in the test-set.
    acc = float(correct_sum) / len(correct)

    return acc, correct_sum

def validation_accuracy():
    # Get the array of booleans whether the classifications are correct
    # for the validation-set.
    # The function returns two values but we only need the first.
    correct, _ = predict_cls_validation()

    # Calculate the calssification accuracy and return it.
    return cls_accuracy(correct)

def optimize(num_iterations):
    # Ensure we update the global variables rather than local copies.
    global total_iterations
    global best_validation_accuracy
    global last_improvement

    # Start-tiem used for printing time-usage below.
    start_time = time.time()

    for i in range(num_iterations):
        # Increase the total number of iterations performed.
        # It is easier to update it in eah iterations because
        # we need this number several times in the flowwing.
        total_iterations += 1

        # Get a batch of training exaples.
        # x_batch now holds a batch of images and
        # y_true-batch are the true labels for those images.
        x_batch, y_true_batch, _ = data.random_batch(train_batch_size)

        # Put the batch into a dict with the proper names
        # for placeholder variables in the TensorFlow graph.
        feed_dict_train = {x: x_batch,
                           y_true: y_true_batch}

        # Run the optimizer using this batch of training data.
        # TensorFlow assigns the variables in feed_dict_train
        # to the placeholder variables and then runs the optimizer.
        session.run(optimizer, feed_dict=feed_dict_train)

        # Print status every 100 iterations and after last iteration.
        if (total_iterations % 100 == 0) or (i == (num_iterations - 1)):
            # Calculate the accuracy on the training-batch.
            acc_train = session.run(accuracy, feed_dict=feed_dict_train)

            # Calculate the accuracy on the validation-set.
            # The function returns 2 values but we only need the first.
            acc_validation, _ = validation_accuracy()

            # If validation accuracy is an improvement over best-know.
            if acc_validation > best_validation_accuracy:
                # Update the best-known validation accuracy.
                best_validation_accuracy = acc_validation

                # Set the iteration for the last improvement to current.
                last_improvement = total_iterations

                # Save all variables of the TensorFlwo graph to file.
                saver.save(sess=session, save_path=save_path)

                # A string to be printed below, shows improvement found.
                improved_str = '*'
            else:
                # An empty string to be printed below.
                # Shows that no improvment was found
                improved_str = ''

            # Status-message for printing.
            msg = "Iter: {0:>6}, Train-Batch Accuracy: {1:>6.1%}, Validation Acc: {2:>6.1%} {3}"

            # Print it.
            print(msg.format(i+1, acc_train, acc_validation, improved_str))

        # If no improvement found in the required number of iterations.
        if total_iterations - last_improvement > require_improvement:
            print("No improvement found in a while, stopping optimization.")

            # Break out from the for-loop.
            break

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
    # all iamges in the test-set.

    # correct is a boolean array whether the prediced class
    # is equal to the true class for each image in the test-set.

    # Negate the boolean arrays.
    incorrect = (correct == False)

    # Get the iamges from the test-set that have been
    # incorrectly classified.
    images_ = data.x_test[incorrect]

    # Get the predicted classes for those images.
    cls_pred = cls_pred[incorrect]

    # Get the true clases for those iamges.
    cls_true_ = data.y_test_cls[incorrect]

    # Plot the first 9 images.
    plot_images(images_=images_[0:9],
                cls_true_=cls_true_[0:9],
                cls_pred=cls_pred[0:9])

# Helper-function to plot confusion matrix
def plot_confusion_matrix(cls_pred):
    # This is called from print_test_accuracy() below.

    # cls_pred is an array of the predicted class-number for
    # all iamges in the test-set.

    # Get the true classifications for the test-set.
    cls_true_ = data.y_test_cls

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

# Helper-functions for calculating classifications
# Split the data-set in batches of this size to limit RAM usage.
batch_size = 256

def predict_cls(images_, labels, cls_true_):
    # Number of images.
    num_images = len(images_)

    # Allocate an array for the predicted classes which
    # will be calculated in batches and filled into this array.
    cls_pred = np.zeros(shape=num_images, dtype=np.uint8)

    # Now calculate the predicted classes for the batches.
    # We will just iterate through all the batches.
    # There might be a more clever and Pythonic way of doing this.

    # The starting index for the next batch is denoted i.
    i = 0

    while i < num_images:
        # The ending index for the next batch is denoted j.
        j = min(i + batch_size, num_images)

        # Create a feed-dict with the images and labels
        # between index i and j.
        feed_dict = {x: images_[i:j, :],
                     y_true: labels[i:j, :]}

        # Calculate the predicted class using TensorFlow.
        cls_pred[i:j] = session.run(y_pred_cls, feed_dict=feed_dict)

        # Set the start-index for the next batch to the
        # end-index of the current batch.
        i = j

    # Create a boolean array whether each image is correctly classified.
    correct = (cls_true_ == cls_pred)

    return correct, cls_pred

def predict_cls_test():
    return predict_cls(images_=data.x_test,
                       labels=data.y_test,
                       cls_true_=data.y_test_cls)

def predict_cls_validation():
    return predict_cls(images_=data.x_val,
                       labels=data.y_val,
                       cls_true_=data.y_val_cls)

# Helper-function for showing the performance
def print_test_accuracy(show_example_errors=False,
                        show_confusion_matrix=False):
    # Fora ll the images in the test-set,
    # calculate the predicted classes and whether they are correct.
    correct, cls_pred = predict_cls_test()

    # Classification accuracy and teh number of correct classifications.
    acc, num_correct = cls_accuracy(correct)

    # Number of images being classified.
    num_images = len(correct)

    # Print the accuracy.
    msg = "Accuracy on Test-set: {0:.1%} ({1} / {2})"
    print(msg.format(acc, num_correct, num_images))

    # Plot some examples of mis-classifications, if desired.
    if show_example_errors:
        print("Example errors:")
        plot_example_errors(cls_pred=cls_pred, correct=correct)

    # Plot the confusion matrix, if desired.
    if show_confusion_matrix:
        print("Confusion Matrix:")
        plot_confusion_matrix(cls_pred=cls_pred)

# Helper-functon for plotting convolutional weights
def plot_conv_weights(weights, input_channel=0):
    # Assume weights are TensorFlow ops for 4-dim variables
    # e.g. weights_conv1 or weighs_conv2

    # Retrieve the values of the weight-variables from TensorFlow.
    # A feed-dict is not necessary because nothing is calculated.
    w = session.run(weights)

    # Print mean and standard deviation.
    print("Mean: {0:.5f}, Stdev: {1:.5f}".format(w.mean(), w.std()))

    # Get the lowest and highest values for the weights.
    # This is used to correct the colour intensity across
    # the images so they can be compared with each other.
    w_min = np.min(w)
    w_max = np.max(w)

    # Number of filters used in conv. layer.
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
            # The format of this 4-dim tensor is determined by the
            # TensorFlow API. See Tutorial #02 for more details.
            img = w[:, :, input_channel, i]

            # Plot image.
            ax.imshow(img, vmin=w_min, vmax=w_max, interpolation='nearest', cmap='seismic')

        # Remove ticks from the plot.
        ax.set_xticks([])
        ax.set_yticks([])

    # Ensure the plot is shown correctly with multiple plots
    # in a single Notebook cell.
    plt.show()

# Performance before any optimization
print("\nPerformance before any optimization!")
print_test_accuracy()
plot_conv_weights(weights=weight_conv1)

# Perform 10,000 optimization iterations
print("\nPerform 10,000 optimization iterations")
optimize(num_iterations=10000)
print_test_accuracy(show_example_errors=True,
                    show_confusion_matrix=True)
plot_conv_weights(weights=weight_conv1)

# Initialize Variables Again
init_variables()
print_test_accuracy()
plot_conv_weights(weights=weight_conv1)

# Restore Best Variables
saver.restore(sess=session, save_path=save_path)
print("\nRestore success!")
print_test_accuracy(show_example_errors=True,
                    show_confusion_matrix=True)
plot_conv_weights(weights=weight_conv1)

# Close TnesorFlow Session
# This has been commented out in case you want to modify and experiment
# with the Notebook without having to restart it.
session.close()
