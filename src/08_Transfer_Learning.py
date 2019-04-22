# Imports
import os
import time
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from datetime import timedelta

# Functions and classes for loading and using the Inception model.
import inception
import cifar10
from cifar10 import num_classes

print('TensorFlow version: {}'.format(tf.__version__))

cifar10.maybe_download_and_extract()
class_names = cifar10.load_class_names()

print('class_names: {}'.format(class_names))

images_train, cls_train, labels_train = cifar10.load_training_data()
images_test, cls_test, labels_test = cifar10.load_test_data()

print("Size of:")
print("- Training-set:\t\t{}".format(len(images_train)))
print("- Test-set:\t\t{}".format(len(images_test)))

# Helper-function for plotting images
def plot_images(images, cls_true, cls_pred=None, smooth=True):
    assert len(images) == len(cls_true)

    # Create figure with sub-plots.
    fig, axes = plt.subplots(3, 3)

    # Adjust vertical spacing.
    if cls_pred is None:
        hspace = 0.3
    else:
        hspace = 0.6

    fig.subplots_adjust(hspace=hspace, wspace=0.3)

    # Interpolation type.
    if smooth:
        interpolation = 'spline16'
    else:
        interpolation = 'nearest'

    for i, ax in enumerate(axes.flat):
        # There may be less than 9 images, ensure it doesn't crash.
        if i < len(images):
            # Plot image.
            ax.imshow(images[i], interpolation=interpolation)

            # Name of the true class.
            cls_true_name = class_names[cls_true[i]]

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

    # Ensure the plot is shown correctly with multiple plots
    # in a single Notebook cell.
    plt.show()

# Plot a few images to see if data is correct
# Get the first images from the test-set.
images = images_test[0:9]

# Get the true classes for those images.
cls_true = cls_test[0:9]

# Plot the images and labels using our helper-function above.
plot_images(images=images, cls_true=cls_true, smooth=True)

# Download the Inception Model
inception.maybe_download()

# Load the Inception Model
model = inception.Inception()

# Calculate Transfer-Values
from inception import transfer_values_cache

file_path_cache_train = os.path.join(cifar10.data_path, 'inception_cifar10_train.pkl')
file_path_cache_test = os.path.join(cifar10.data_path, 'inception_cifar10_test.pkl')

print("Processing Inception transfer-values for training-images ...")

# Scale images because Inception needs pixels to be between 0 and 255,
# while the CIFAR-10 functions return pixels between 0.0 and 1.0
images_scaled = images_train * 255.0

# If transfer-values have already been calculated then reload them,
# otherwise calculate them and save them to a cache-file.
transfer_values_train = transfer_values_cache(cache_path=file_path_cache_train,
                                              images=images_scaled,
                                              model=model)

print("Processing Inception transfer-values for test-images ...")

# Scale images because Inception need pixels to be between 0 and 255,
# while the CIFAR-10 functions return pixels between 0.0 and 1.0
images_scaled = images_test * 255.0

# If transfer-values have already been calculated then reload them,
# otherwise calculate them and save them to a cache-file.
transfer_values_test = transfer_values_cache(cache_path=file_path_cache_test,
                                             images=images_scaled,
                                             model=model)

print('transfer_values_train.shape: {}'.format(transfer_values_train.shape))
print('transfer_values_test shape: {}'.format(transfer_values_test.shape))

# Helper-function for plotting transfer-values
def plot_transfer_values(i):
    print("Input image:")

    # Plot the i'th image from the test-set.
    plt.imshow(images_test[i], interpolation='nearest')
    plt.show()

    print("Transfer-values for the image using Inception model:")

    # Transform the ransfer-values into an image.
    img = transfer_values_test[i]
    img = img.reshape((32, 64))

    # Plot the image for the transfer-values.
    plt.imshow(img, interpolation='nearest', cmap='Reds')
    plt.show()

plot_transfer_values(i=16)
plot_transfer_values(i=17)

# Analysis of Transfer-Values using PCA
from sklearn.decomposition import PCA

pca = PCA(n_components=2)

transfer_values = transfer_values_train[0:3000]
cls = cls_train[0:3000]
print('transfer_values.shape: {}'.format(transfer_values.shape))

transfer_values_reduced = pca.fit_transform(transfer_values)
print('PCA transfer_values_reduced.shape: {}'.format(transfer_values_reduced.shape))

def plot_scatter(values, cls):
    # Create a color-map with a different color for each class.
    import matplotlib.cm as cm
    cmap = cm.rainbow(np.linspace(0.0, 1.0, num_classes))

    # Get the color for each sample.
    colors = cmap[cls]

    # Extract the x- and y-values.
    x = values[:, 0]
    y = values[:, 1]

    # Plot it.
    plt.scatter(x, y, color=colors)
    plt.show()

plot_scatter(transfer_values_reduced, cls)

# Analysis of Transfer-Values using t-SNE
from sklearn.manifold import TSNE

# Another method for doing dimensionality reduction is t-SNE. Unfortunately, t-SNE is very slow so we first use PCA
# to reduce the transfer-values from 2048 to 50 elements.

pca = PCA(n_components=50)
transfer_values_50d = pca.fit_transform(transfer_values)

tsne = TSNE(n_components=2)
transfer_values_reduced = tsne.fit_transform(transfer_values_50d)
print('t-SNE transfer_values_reduced.shape: {}'.format(transfer_values_reduced.shape))

plot_scatter(transfer_values_reduced, cls)

# New Classifier in TensorFlow
# Placeholder Variables
transfer_len = model.transfer_len
x = tf.placeholder(tf.float32, shape=[None, transfer_len], name='x')
y_true = tf.placeholder(tf.float32, shape=[None, num_classes], name='y_true')
y_true_cls = tf.math.argmax(y_true, axis=1)

# Neural Network
layer_fc1 = tf.layers.dense(inputs=x, units=1024, activation=tf.nn.relu, name='layer_fc1')
y_pred = tf.layers.dense(inputs=layer_fc1, units=num_classes, activation=None, name='layer_pred')
loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=y_pred, labels=y_true)

# Optimization Method
global_step = tf.get_variable(initializer=0, trainable=False, name='global_step')
optimizer = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(loss, global_step)




# Classification Accuracy

# TensorFlow Run
# Create TensorFlow Session

# Initialize Variables

# Helper-function to get a random training-batch

# Helper-function to perform optimization

# Helper-function for Showing Results
# Helper-function to plot example errors

# Helper-function to plot confusion matrix

# Helper-functions for calculating classifications

# Helper-function for calculating classification accuracy

# Helper-function for showing the classification accuracy

# Results
# Performance before any optimization

# Performance after 10,000 optimization iterations

# Close TensorFlow Session
