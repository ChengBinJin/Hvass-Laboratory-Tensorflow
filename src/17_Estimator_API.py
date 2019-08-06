# Introduction
# Imports
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

print("TensorFlow version: {}".format(tf.__version__))

# Load Data
from mnist import MNIST
data = MNIST(data_dir="../data/MNIST/")

print("Size of:")
print("- Training-set:\t\t{}".format(data.num_train))
print("- Validation-set:\t{}".format(data.num_val))
print("- Test-set:\t\t{}".format(data.num_test))

# The number of pixels in each dimension of an image.
img_size = data.img_size

# The images are stored in one-dimensinal arrays of this length.
img_size_flat = data.img_size_flat

# Tuple with heigh and width of iamges used to reshape arrays.
img_shape = data.img_shape

# Number of classes, one class for each of 10 digits.
num_classes = data.num_classes

# Number of colour channels for the images: 1 channel for gray-scale.
num_channel = data.num_channels

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

# Plot the iamges and labels using our helper-function above.
plot_images(images=images, cls_true=cls_true)

# Input Functions for the Estimator
train_input_fn = tf.estimator.inputs.numpy_input_fn(
    x={"x": np.array(data.x_train)},
    y=np.array(data.y_train_cls),
    num_epochs=None,
    shuffle=True)

test_input_fn = tf.estimator.inputs.numpy_input_fn(
    x={"x": np.array(data.x_test)},
    y=np.array(data.y_test_cls),
    num_epochs=1,
    shuffle=False)

some_images = data.x_test[0:9]
predict_input_fn = tf.estimator.inputs.numpy_input_fn(
    x={"x": some_images},
    num_epochs=1,
    shuffle=False)

some_images_cls = data.y_test_cls[0:9]