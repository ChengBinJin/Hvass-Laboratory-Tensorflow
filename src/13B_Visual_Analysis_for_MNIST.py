# TensorFlow Tutorial #13-B
# Visual Analysis (MNIST)

# Introduction
# Flowchart
# Imports
import math
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.metrics import confusion_matrix

from mnist import MNIST

print('TensorFlow version: {}'.format(tf.__version__))

# Load Data
data = MNIST(data_dir='../data/MNIST')

print("Size of:")
print("- Training-set:\t\t{}".format(data.num_train))
print("- Validation-set:\t{}".format(data.num_val))
print("- Test-set:\t\t{}".format(data.num_test))

# The number of pixels in each dimension of an image.
img_size = data.img_size

# The images are stored in one-dimensional arrays of this length.
img_size_flat = data.img_size_flat

# Tuple with height and width of images used to reshape arrays.
img_shape = data.img_shape

# Number of classes, one class for each of 10 digits.
num_classes = data.num_classes

# Number of colour channels for the images: 1 channel for gray-scale
num_channels = data.num_channels

# Helper-functions for plotting images
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


def plot_images10(images, smooth=True):
    # Interpolation type.
    if smooth:
        interpolation = 'spline16'
    else:
        interpolation = 'nearest'

    # Create figure with sub-plots.
    fig, axes = plt.subplots(2, 5)

    # Adjust vertical spacing.
    fig.suplots_adjust(hspace=0.1, wspace=0.1)

    # For each entry in the grid.
    for i, ax in enumerate(axes.flat):
        # Get the i'th image and only use the desired pixels.
        img = images[i, :, :]

        # Plot the image
        ax.imshow(img, interpolation=interpolation, cmap='binary')

        # Remove ticks.
        ax.set_xticks([])
        ax.set_yticks([])

    # Ensure the plot is shown correctly with multiple plots
    # in a single Notebook cell.
    plt.show()


def plot_image(image):
    plt.imshow(image, interpolation='nearest', cmap='binary')
    plt.xticks([])
    plt.yticks([])

# Plot a few images to see if data is correct
# Get hte first images from the test-set.
indexs = np.random.random_integers(low=0, high=data.num_test, size=9)
images = [data.x_test[index] for index in indexs]

# Get the true classes for the those images.
cls_true = [data.y_test_cls[index] for index in indexs]

# Plot the images and labels using our helper-function above.
plot_images(images=images, cls_true=cls_true)


# TensorFlow Graph
# Placeholder variables
x = tf.placeholder(dtype=tf.float32, shape=[None, img_size_flat], name='x')
y_true = tf.placeholder(dtype=tf.float32, shape=[None, num_classes], name='y_true')

x_image = tf.reshape(tensor=x, shape=[-1, img_size, img_size, num_channels])
y_true_cls = tf.argmax(y_true, axis=1)

# Neural Network
net = x_image
net = tf.layers.conv2d(inputs=net, name='layer_conv1', padding='same', filters=16, kernel_size=5, activation=tf.nn.relu)
net = tf.layers.max_pooling2d(inputs=net, pool_size=2, strides=2)

net = tf.layers.conv2d(inputs=net, name='layer_conv2', padding='same', filters=36, kernel_size=5, activation=tf.nn.relu)
net = tf.layers.max_pooling2d(inputs=net, pool_size=2, strides=2)
net = tf.layers.flatten(inputs=net)

net = tf.layers.dense(inputs=net, name='layer_fc1', units=128, activation=tf.nn.relu)
net = tf.layers.dense(inputs=net, name='layer_fc_out', units=num_classes, activation=None)

logits = net
y_pred = tf.nn.softmax(logits=logits)
y_pred_cls = tf.argmax(y_pred, axis=1)

# Loss-Function to be Optimized
cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(labels=y_true, logits=logits)
loss = tf.reduce_mean(cross_entropy)

# Optimization Method
optimizer = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(loss)

# Classification Accuracy
correct_prediction = tf.equal(y_pred_cls, y_true_cls)
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# Optimize the Neural Network
# Create TensorFlow session
session = tf.Session()

# Initialize variables
session.run(tf.global_variables_initializer())

# Helper-function to perform optimization iterations
train_batch_size = 64

# Counter for total number of iterations performed so far.
total_iterations = 0

def optimize(num_iterations):
    # Ensure we update the global variable rather than a local copy.
    global total_iterations

    for i in range(total_iterations, total_iterations + num_iterations):
        # Get a batch of training examples.
        # x_batch now holds a batch of images and
        # y_true_batch are the true labels for those names.
        x_batch, y_true_batch, _ = data.random_batch(batch_size=train_batch_size)

        # Put the batch into a dict with the proper names
        # for placeholder variables in the TensorFlow graph.
        feed_dict_train = {x: x_batch,
                           y_true: y_true_batch}

        # Run the optimizer using this batch of training data.
        # TensorFlow assigns the varabiels in feed_dict_train
        # to the placeholder variables and then rns the optimizer.
        session.run(optimizer, feed_dict=feed_dict_train)

        # Run the optimizer using this batch of training data.
        # TensorFlow assigns the variables in feed_dict_train
        # to the placeholder variables and then runs the optimizer.
        session.run(optimizer, feed_dict=feed_dict_train)

        # Print status every 100 iterations
        if i % 100 == 0:
            # Calculate the accuracy on the training-set.
            acc = session.run(accuracy, feed_dict=feed_dict_train)

            # Message for printing.
            msg = "Optimization Iterations: {0:>6}, Training Accuracy: {1:>6.1%}"

            # Print it.
            print(msg.format(i + 1, acc))

        # Update the total number of iterations performed.
        total_iterations += num_iterations

# Helper-function to plot example errors

# Helper-function to plot confusion matrix

# Helper-function for showing the performance



# Performance before any optimization

# Performance after 10,000 optimization iterations

# Optimizing the Input Images



# Helper-function for getting the names of convolutional layers

# Helper-function for finding the input image

# First Convolutional Layer



# Second Convolutional Layer

# Final Output Layer

# Close TensorFlow Session

