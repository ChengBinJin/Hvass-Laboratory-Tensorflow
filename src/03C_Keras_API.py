import math
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
# import prettytensor as pt

# from tf.keras.models import Sequential  # This does not work!
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import InputLayer, Input
from tensorflow.python.keras.layers import Reshape, MaxPooling2D
from tensorflow.python.keras.layers import Conv2D, Dense, Flatten
# from tensorflow.python.keras.layers import Adam

from mnist import MNIST

print('TensorFlow version: {}'.format(tf.__version__))

data = MNIST(data_dir="../data/MNIST")

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

# Tuple with height, width and depth used to reshape arrays.
# This is used for reshaping in Keras.
img_shape_full = data.img_shape_full

# Number of classes, one class for each of 10 digits.
num_classes = data.num_classes

# Number of colour channels for the images: 1 channel for gray-scale.
num_channels = data.num_channels

# Helper-function for plotting images
def plot_images(images_, cls_true_, cls_pred=None):
    assert len(images_) == len(cls_true_) == 9

    # Create figure with 3x3 sub-plots.
    fig, axes = plt.subplots(3, 3)
    fig.subplots_adjust(hspace=0.3, wspace=0.3)

    for i, ax in enumerate(axes.flat):
        # Plot image.
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
# Get the first images from the test-set
images = data.x_test[0:9]

# Get the true classes for those images.
cls_true = data.y_test_cls[0:9]

# Plot the images and labels using our helper-function above.
plot_images(images_=images, cls_true_=cls_true)

# Helper-function to plot example errors
def plot_example_errors(cls_pred):
    # cls_pred is an array of the predicted class-number for
    # all images in the test-set.

    # Boolean array whether the predicted class is incorrect.
    incorrect = (cls_pred != data.y_test_cls)

    # Get the iamges from the test-set that have been
    # incorrrectly classified.
    images_ = data.x_test[incorrect]

    # Get the predicted classes for those images.
    cls_pred = cls_pred[incorrect]

    # Get the true classes for those images.
    cls_true_ = data.y_test_cls[incorrect]

    # Plot the first 9 images.
    plot_images(images_=images_[0:9],
                cls_true_=cls_true_[0:9],
                cls_pred=cls_pred[0:9])

# if False:
#     x_pretty = pt.wrap(x_image)
#
#     with pt.defaults_scope(activation_fn=tf.nn.relu):
#         y_pred, loss = x_pretty.\
#             conv2d(kernel=5, depth=16, name='layer_conv1').\
#             max_pool(kernel=2, stride=2).\
#             conv2d(kernel=5, depth=36, name='layer_conv2').\
#             max_pool(kernel=2, stride=2).\
#             flatten().\
#             fully_connected(size=128, name='layer_fc1').\
#             softmax_classifier(num_classes=num_classes, labels=y_true)

# Sequential Model
# Start construction of the Keras Sequential model.
model = Sequential()

# Add an input layer which is similar to a feed_dict in TensorFlow.
# Note that the input-shape must be a tuple containing the image-size.
model.add(InputLayer(input_shape=(img_size_flat,)))

# The input is a flattened array awith 784 elements,
# but the convolutional layers expect images with shape (28, 28, 1)
model.add(Reshape(img_shape_full))

# First convolutional layer with ReLU-activation and max-pooling.
model.add(Conv2D(kernel_size=5, strides=1, filters=16, padding='same', activation='relu', name='layer_conv1'))
model.add(MaxPooling2D(pool_size=2, strides=2))

# Second convolutional layer with ReLU-activation and max-pooling.
model.add(Conv2D(kernel_size=5, strides=1, filters=36, padding='same', activation='relu', name='layer_conv2'))
model.add(MaxPooling2D(pool_size=2, strides=2))

# Flatten the 4-rank output of the convolutional layers
# to 2-rank that can be input to a fully-connected / dense layer.
model.add(Flatten())

# First full-connected / dense layer with ReLU-activation.
model.add(Dense(128, activation='relu'))

# Last fully-connected / dense layer with softmax-activation
# for use in classification.
model.add(Dense(num_classes, activation='softmax'))

# Model Compilation
optimizer = tf.keras.optimizers.Adam(lr=1e-3)
model.compile(optimizer=optimizer,
               loss='categorical_crossentropy',
               metrics=['accuracy'])

# Training
model.fit(x=data.x_train,
          y=data.y_train,
          epochs=1, batch_size=128)

# Evaluation
result = model.evaluate(x=data.x_test,
                        y=data.y_test)

for name, value in zip(model.metrics_names, result):
    print(name, value)

print("{0}: {1:.2%}".format(model.metrics_names[1], result[1]))

# Examples of Mis-Classified Images

