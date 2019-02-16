import math
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import prettytensor as pt
from sklearn.metrics import confusion_matrix

# from tensorflow.examples.tutorials.mnist import input_data
from mnist import MNIST

print("TensorFlow version: {}".format(tf.__version__))

# Load Data
data = MNIST(data_dir="../data/MNIST/")
data_test_cls = data.y_test_cls

print("Size of:")
print("- Training-set:\t\t{}".format(data.num_train))
print("- Validation-set:\t{}".format(data.num_val))
print("- Test-set:\t\t{}".format(data.num_test))

# Data Dimensions
# We know that MNISt images are 28 pixels in each dimension.
img_size = 28

# Images are stored in one-dimensional arrays of this length.
img_size_flat = img_size * img_size

# Tuple with height and width of images used to reshape arrays.
img_shape = (img_size, img_size)

# Number of colour channels for the images: 1 channel for gray-scale.
num_channels = 1

# Number of classes, one class for each of 10 digits.
num_classes = 10

# Helper-function for plotting images
def plot_images(images, cls_true, cls_pred=None):
    assert len(images) == len(cls_true) == 9

    # Create figure with 3x3 sub-plots.
    fig, axes = plt.subplots(3, 3)
    fig.subplots_adjust(hspace=0.3, wspace=0.3)

    for i, ax in enumerate(axes.flat):
        # Plot image.
        ax.imshow(images[i].reshape(img_shape), cmap='binary')

        # Show true and predicted classes.
        if cls_pred is None:
            xlabel = "True: {0}".format(cls_true[i])
        else:
            xlabel = "True: {0}, Pred: {1}".format(cls_true[i], cls_pred[i])

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
plot_images(images=images, cls_true=cls_true)

# TensorFlow Graph
# Placeholder variables
x = tf.placeholder(tf.float32, shape=[None, img_size_flat], name='x')
x_image = tf.reshape(x, [-1, img_size, img_size, num_channels])

y_true = tf.placeholder(tf.float32, shape=[None, num_classes], name='y_true')
y_true_cls = tf.argmax(y_true, dimension=1)

is_run = False
if is_run:
    x_pretty = pt.wrap(x_image)

    with pt.defaults_scope(activation_fn=tf.nn.relu):
        y_pred_, loss_ = x_pretty.conv2d(kernel=5, depth=16, name='layer_conv1').\
            max_pool(kernel=2, stride=2).\
            conv2d(kernel=5, depth=36, name='layer_conv2').\
            max_pool(kernel=2, stride=2).\
            flatten().\
            fully_connected(size=128, num='layer_fc1').\
            softmax_classifier(num_classes=num_classes, labels=y_true)

# Layers Implementation
net = x_image
net = tf.layers.conv2d(inputs=net, name='layer_conv1', padding='same', filters=16, kernel_size=5, activation=tf.nn.relu)
layer_conv1 = net

net = tf.layers.max_pooling2d(inputs=net, pool_size=2, strides=2)
layer_max1 = net

net = tf.layers.conv2d(inputs=net, name='layer_conv2', padding='same', filters=36, kernel_size=5, activation=tf.nn.relu)
layer_conv2 = net

net = tf.layers.max_pooling2d(inputs=net, pool_size=2, strides=2)
layer_max2 = net
print('net: {}'.format(net))

net = tf.contrib.layers.flatten(net)
# This should be eventually be replaced by:
# net = tf.layers.flatten(net)
print('net: {}'.format(net))

net = tf.layers.dense(inputs=net, name='layer_fc1', units=128, activation=tf.nn.relu)
net = tf.layers.dense(inputs=net, name='layer_fc_out', units=num_classes, activation=None)

logits = net
y_pred = tf.nn.softmax(logits=logits)
y_pred_cls = tf.argmax(y_pred, dimension=1)

# Loss-Function to be optimized
cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(labels=y_true, logits=logits)
loss = tf.reduce_mean(cross_entropy)

# Optimization Method
optimizer = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(loss)

# Classification Accuracy
correct_prediction = tf.equal(y_pred_cls, y_true_cls)
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

for var in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES):
    print(var)

def get_weights_variable(layer_name):
    # Retrieve an existing variable named 'kernel' in the scope
    # wit the given layer_name.
    # This is awkward because the TensorFlow function was
    # really intended for another purpose.
    with tf.variable_scope(layer_name, reuse=True):
        variable = tf.get_variable('kernel')

    return variable

weights_conv1 = get_weights_variable(layer_name='layer_conv1')
weights_conv2 = get_weights_variable(layer_name='layer_conv2')

# TensorFlow Run
# Create TensorFlow Session
session = tf.Session()

# Initialize variables
session.run(tf.global_variables_initializer())

# Helper-function to perform optimization iterations
train_batch_size = 64

# Cpimter fpr total number of iterations performed so far
total_iterations = 0

def optimize(num_iterations):
    # Ensure we update the global variables rather than a local copy.
    global total_iterations

    for i in range(total_iterations,
                   total_iterations + num_iterations):

        # Get a batch of training examples.
        # x_batch now holds a batch of images and
        # y_true_batch are the true labels for those images.
        x_batch, y_true_batch, _ = data.random_batch(train_batch_size)

        # Put the batch into a dict with the proper names
        # for placeholder variables in the TensorFlow graph.
        feed_dict_train = {x: x_batch,
                           y_true: y_true_batch}

        # Run the optimizer using this batch of training data.
        # TensorFlow assigns the variables in feed_dict_train
        # to the placeholder variables and then runs the optimizer.
        session.run(optimizer, feed_dict=feed_dict_train)

        # Print status every 100 iterations.
        if i % 100 == 0:
            # Calculate the accuracy on the training-set.
            acc = session.run(accuracy, feed_dict=feed_dict_train)

            # Message for printing.
            msg = "Optimization Iteration: {0:>6}, Training Accuracy: {1:>6.1%}"

            # Print it.
            print(msg.format(i+1, acc))

    # Update the total number of iterations performed.
    total_iterations += num_iterations

# Helper-function to plot example errors
def plot_example_errors(cls_pred, correct):
    # This function is called from print_test_accuracy() below.

    # cls_pred is an array of the predicted class-number for
    # all images in the test-set.

    # correct is a boolean array whether the predicted class
    # is equal to the true class for each image in the test-set.

    # Negate the boolean array.
    incorrect = (correct == False)

    # Get the iamges from the test-set that have been
    # incorrectly classified.
    images_ = data.x_test[incorrect]

    # Get the predicted classes for those images.
    cls_pred = cls_pred[incorrect]

    # Get the predicted classes for those images.
    cls_true_ = data.y_test_cls[incorrect]

    # Plot the first 9 images.
    plot_images(images=images_[0:9],
                cls_true=cls_true_[0:9],
                cls_pred=cls_pred[0:9])

# Helper-function to plot confusion matrix
def plot_confusion_matrix(cls_pred):
    # This is called from print_test_accuracy() below.

    # cls_pred is an array of the predicted class-number for
    # all images in the test-set.

    # Get the true classifications for the test-set.
    cls_true_ = data.y_test_cls

    # Get the confusion matrix using sklearn.
    cm = confusion_matrix(y_true=cls_true_, y_pred=cls_pred)

    # Print the confusion matrix as text.
    print(cm)

    # plot the confusion matrix as an image.
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
# Split the test-set int osmaller batches of this size.
test_batch_size = 256

def print_test_accuracy(show_example_errors=False,
                        show_confusion_matrix=False):
    # Number of images in the test-set.
    num_test = len(data.y_test_cls)

    # Allocate an array for the predicted clases which
    # will be calculated in batches and filled into this array.
    cls_pred = np.zeros(shape=num_test, dtype=np.int)

    # Now calculate the predicted classes for the batches.
    # We will just iterate through all the batches.
    # There might be a more clever and Pythonic way of doing this.

    # The starting index for the next batch is denoted i.
    i = 0

    while i < num_test:
        # There ending index for the next batch is denoted j.
        j = min(i + test_batch_size, num_test)

        # Get the images from the test-set between index i and j.
        images_ = data.x_test[i:j, :]

        # Get the associated labels.
        labels = data.y_test[i:j]

        # Create a feed-dict with these images and labels.
        feed_dict = {x: images_,
                     y_true: labels}

        # Calculate the predicted class using TensorFlow.
        cls_pred[i:j] = session.run(y_pred_cls, feed_dict=feed_dict)

        # Set the start-index for the next batch to the
        # end-index of the current batch.
        i = j

    # Convenience variable for the true class-numbers of the test-set.
    cls_true_ =data.y_test_cls

    # Create a boolean array whether each image is correctly classified.
    correct = (cls_true_ == cls_pred)

    # Calculate the number of corectly classified images.
    # When summing a boolean array, False means 0 and True means 1.
    correct_sum = correct.sum()

    # Classification accuracy is the number of correctly classified
    # images divided by the total number of images in the test-set
    acc = float(correct_sum) / num_test

    # Print the accuracy.
    msg = "Accuracy on Test-Set: {0:.1%} ({1} / {2})"
    print(msg.format(acc, correct_sum ,num_test))

    # Plot some examples of mis-classifications, if desired.
    if show_example_errors:
        print("Example errors:")
        plot_example_errors(cls_pred=cls_pred, correct=correct)

    # Plot the confusion matrix, if desired.
    if show_confusion_matrix:
        print("Confusion Matrix:")
        plot_confusion_matrix(cls_pred=cls_pred)

# Performance before optimization
print_test_accuracy()

# Performance after 1 optimization iteration
optimize(num_iterations=1)
print_test_accuracy()

# Performance after 100 optimization iterations
optimize(num_iterations=99)  # We already performed 100 iteration above.
print_test_accuracy(show_example_errors=True)

# Performance after 1000 optimization iterations
optimize(num_iterations=900)  # We performed 1000 iterations above.
print_test_accuracy(show_example_errors=True)

# Performance after 10,000 optimization iterations
optimize(num_iterations=9000)  # We performed 10,000 iterations above.
print_test_accuracy(show_example_errors=True,
                    show_confusion_matrix=True)

# Visualization of Weights and Layers
# Helper-function for plotting convolutional weights
def plot_conv_weights(weights, input_channel=0):
    # Assume weights are TensorFlow ops for 4-dim variables
    # e.g. weights_conv1 or weights_conv2

    # Eetrieve the values of the weight-variables from TensorFlow.
    # A feed-dict is not necessary because nothing is calculated.
    w = session.run(weights)

    # Get the lowest and highest values for the weights.
    # This is used to correct the colour intensity across
    # the images so they can be compared with each oterh.
    w_min = np.min(w)
    w_max = np.max(w)

    # Number of filters used in the conv. layer.
    num_filters = w.shape[3]

    # Number of grids to plot.
    # Rounded-up. square-root of the number of filters.
    num_grids = math.ceil(math.sqrt(num_filters))

    # Create figure with a grid of sub-plots.
    fig, axes = plt.subplots(num_grids, num_grids)

    # Plot all the filter-weights
    for i, ax in enumerate(axes.flat):
        # Only plot the valid filter-weights.
        if i < num_filters:
            # Get the weights for the i't h filter of the input channel.
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

# Helper-function for plotting the output of a convolutional layer
def plot_conv_layer(layer, image):
    # Assume layer is a TensorFlow op that outputs a 4-dim tensor
    # which is the outptu of a convolutional layer,
    # e.g. layer_conv1 or layer_conv2.

    # Create a feed-dict containinig just one image.
    # Note that we don't need to feed y_true because it is
    # not used in this calculation.
    feed_dict = {x: [image]}

    # Calculate and retrieve the output values of the layer
    # when inputting that image.
    values = session.run(layer, feed_dict=feed_dict)

    # Number of filters used in the conv. layer.
    num_filters = values.shape[3]

    # Number of grids to plot.
    # Rounded-up, square-root of the number of filters.
    num_grids = math.ceil(math.sqrt(num_filters))

    # Create figure with a grid of sub-plots.
    fig, axes = plt.subplots(num_grids, num_grids)

    # Plot the output images of all the filters.
    for i, ax in enumerate(axes.flat):
        # Only plot the images for valid filters.
        if i < num_filters:
            # Get the output image of using the i'th filter.
            img = values[0, :, :, i]

            # Plot image.
            ax.imshow(img, interpolation='nearest', cmap='binary')

        # Remove ticks from the plot.
        ax.set_xticks([])
        ax.set_yticks([])

    # Ensure the plot is shown correctly with multiple plots
    # in a single Notebook cell.
    plt.show()

# Input Images
def plot_image(image):
    plt.imshow(image.reshape(img_shape),
               interpolation='nearest',
               cmap='binary')

    plt.show()

image1 = data.x_test[0]
plot_image(image1)

image2 = data.x_test[13]
plot_image(image2)

# Convolution Layer 1
plot_conv_weights(weights_conv1)
plot_conv_layer(layer=layer_conv1, image=image1)
plot_conv_layer(layer=layer_max1, image=image1)
plot_conv_layer(layer=layer_conv1, image=image2)
plot_conv_layer(layer=layer_max1, image=image2)

# Convolution Layer 2
plot_conv_weights(weights_conv2, input_channel=0)
plot_conv_weights(weights_conv2, input_channel=1)
plot_conv_layer(layer=layer_conv2, image=image1)
plot_conv_layer(layer=layer_max2, image=image1)
plot_conv_layer(layer=layer_conv2, image=image2)
plot_conv_layer(layer=layer_max2, image=image2)

# This has been commented out in case you want to modify and experiment
# with the Notebook without having to resart it.
# session.close()
