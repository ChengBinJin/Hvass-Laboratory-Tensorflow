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
    fig.subplots_adjust(hspace=0.1, wspace=0.1)

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
def plot_example_errors(cls_pred, correct):
    # This function is called from print_test_accuracy() below.

    # cls_pred is an array of the predicted class-number for
    # all images in the test-set.

    # correct is a boolean array whether the predicted class
    # is equal to true class for each image in the test-set.

    # Negate the boolean array.
    incorrect = (correct == False)

    # Get the images from the test-set that have been
    # incorrectly classified.
    images = data.x_test[incorrect]

    # Get the predicted classes for those images
    cls_pred = cls_pred[incorrect]

    # Get the true classes for those images.
    cls_true = data.y_test_cls[incorrect]

    # Plot the first 9 images.
    plot_images(images=images[0:9],
                cls_true=cls_true[0:9],
                cls_pred=cls_pred[0:9])

# Helper-function to plot confusion matrix
def plot_confusion_matrix(cls_pred):
    # This is called from print_test_accuracy() below.

    # cls_pred is an array of the predicted class-number for
    # all images in the test-set.

    # Get the true classifications for the test-set.
    cls_true = data.y_test_cls

    # Get the confusion matrix using sklearn.
    cm = confusion_matrix(y_true=cls_true,
                          y_pred=cls_pred)

    # Print the confusion matrix as an image.
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
    num_test = data.num_test

    # Allocate an array for the predicted classes which
    # will be calculated in batches and filled into this array.
    cls_pred = np.zeros(shape=num_test, dtype=np.int)

    # Now calculate the predicted classes for the batches.
    # We will just iterate through all the batches.
    # There might be a more clever and Pythonic way of doing this.

    # The starting index for the next batch is denoted i.
    i = 0

    while i < num_test:
        # The ending index for the next batch is denoted j.
        j = min(i + test_batch_size, num_test)

        # Get the images from the test-set between index i and j.
        images = data.x_test[i:j, :]

        # Get the associated labels.
        labels = data.y_test[i:j, :]

        # Create a feed-dict with these images and labels.
        feed_dict = {x: images,
                     y_true: labels}

        # Calculate the predicted class using TensorFlow
        cls_pred[i:j] = session.run(y_pred_cls, feed_dict=feed_dict)

        # Set the start-index for the next batch to the
        # end-index of the current batch.
        i = j

    # Convenience variable for the true class-numbers of the test-set.
    cls_true = data.y_test_cls

    # Create a boolean array whether each image is correctly classified.
    correct = (cls_true == cls_pred)

    # Calculate the number of correctly classified images.
    # When summing a boolean array, False means 0 and True mean 1.
    correct_sum = correct.sum()

    # Classification accuracy is the number of correctly classified
    # images divided by the total number of images in the test-set.
    acc = float(correct_sum) / num_test

    # Print the accuracy.
    msg = "Accuracy on Test-set: {0:.1%} ({1} / {2})"
    print(msg.format(acc, correct_sum, num_test))

    # Plot some examples of mis-classifications, if desired.
    if show_example_errors:
        print("Example errors:")
        plot_example_errors(cls_pred=cls_pred, correct=correct)

    # Plot the confusion matrix, if desired
    if show_confusion_matrix:
        print("Confusion Matrix:")
        plot_confusion_matrix(cls_pred=cls_pred)

# Performance before any optimization
print_test_accuracy()

# Performance after 10,000 optimization iterations
optimize(num_iterations=10000)
print_test_accuracy(show_example_errors=True,
                    show_confusion_matrix=True)

# Optimizing the Input Images
# Helper-function for getting the names of convolutional layers
def get_conv_layer_names():
    graph = tf.get_default_graph()

    # Create a list of names for the operations in the graph
    # for the Inception model where the operator-type is 'Conv2D'.
    names = [op.name for op in graph.get_operations() if op.type=='Conv2D']

    return names

conv_names = get_conv_layer_names()
print(conv_names)
print('len(conv_names): {}'.format(len(conv_names)))

# Helper-function for finding the input image
def optimize_image(conv_id=None, feature=0, num_iterations=30, show_progress=True):
    # Find an image that maximizes the feature
    # given by the conv_id and feature number.
    # Parameters:   Integer identifying the convolutional layer to
    #               maximize. It is an index into conv_names.
    #               If None then use the last fully-connected layer
    #               before the softmax output.
    # feature:      Index into the layer for the feature to maximize.
    # num_iteration:Number of optimization iterations to perform.
    # show_progess: Boolean whether to show the progress.

    # Create the loss-function that must be maximized.
    if conv_id is None:
        # If we want to maximize a feature on the last layer,
        # then we use the fully-connected layer prior to the
        # softmax-classifier. The feature no. is the class-nuber
        # and must be an integer between 1 and 1000.
        # The loss-function is just the value of that feature.
        loss = tf.math.reduce_mean(input_tensor=logits[:, feature])
    else:
        # If instead we want to maximize a feature of a
        # convolutional layer inside the neural network.

        # Get the name of the convolutional operator.
        conv_name = conv_names[conv_id]

        # Get the default TensorFlow graph.
        graph = tf.get_default_graph()

        # Get a reference to the tensor that is output by the
        # operator. Note that ":0" is added to the name for this.
        tensor = graph.get_tensor_by_name(conv_name + ":0")

        # The loss-function is the average of all the
        # tensor-values for the given feature. This
        # ensures that we generate the whole input image.
        # you can try and modify this so it only uses
        # a part of the tensor.
        loss = tf.reduce_mean(tensor[:, :, :, feature])

    # Get the gradient for the loss-function with regard to
    # the input image. This creates a mathematical
    # function for calculating the gradient.
    gradient = tf.gradients(ys=loss, xs=x_image)

    # Generate a random image of the same size as the raw input.
    # Each pixel is a small random value between 0.45 and 0.55,
    # which is the middle of the valid range between 0 and 1.
    image = 0.1 * np.random.uniform(size=img_shape) + 0.45

    # Perform a number of optimization iterations to find
    # the image that maximizes the loss-function.
    for i in range(num_iterations):
        # Reshape the array so it is a 4-rank tensor.
        img_reshaped = image[np.newaxis, :, :, np.newaxis]

        # Create a feed-dict for inputting the image to the graph.
        feed_dict = {x_image: img_reshaped}

        # Calculate the predicted class-scores,
        # as well as the gradient and the loss-value.
        pred, grad, loss_value = session.run([y_pred, gradient, loss],
                                             feed_dict=feed_dict)

        # Squeeze the dimensionality for the gradient-array.
        grad = np.array(grad).squeeze()

        # The gradient now teels us how much we need to change the
        # input image in order to maximize the given feature.

        # Calculate the step-size for updating the image.
        # This step-size was found to give fast convergence.
        # The addition of 1e-8 is to protect from div-by-zero.
        step_size = 1.0 / (grad.std() + 1e-8)

        # Update the iamge by adding the scaled gradient
        # This is called gradient ascent.
        image += step_size * grad

        # Ensure all pixel-values in the image are between 0 and 1.
        image = np.clip(image, 0.0, 1.0)

        if show_progress:
            print("Iteration: ", i)

            # Convert the predicted class-scores to a on-dim array.
            pred = np.squeeze(pred)

            # The predicted class for the Inception model.
            pred_cls = np.argmax(pred)

            # The score (probability) for the predicted class.
            cls_score = pred[pred_cls]

            # Print the predicted score etc.
            msg = "Predicted class: {0}, score: {1:>7.2%}"
            print(msg.format(pred_cls, cls_score))

            # Print statistics for the gradients
            msg = "Gradient min: {0:>9.6f}, max: {1:>9.6f}, stepsize: {2:9.2f}"
            print(msg.format(grad.min(), grad.max(), step_size))

            # Print the loss-value.
            print("Loss:", loss_value)

            # Newline.
            print()

    return image.squeeze()

# First Convolutional Layer
def optimize_images(conv_id=None, num_iterations=30):
    # Find 10 images that maximize the 10 first features in the layer
    # given by the conv_id
    # Parameters:
    # Conv_id:      Integer identifying the convolutional layer to
    #               maximize. It is an index into conv_names.
    #               If None then use the last layer before the softmax output.
    # num_iterations:   Number of optimization iterations to perform.

    # Which laye are we using?
    if conv_id is None:
        print("Final fully-connected layer before softmax.")
    else:
        print("Layer: ", conv_names[conv_id])

    # Initialize the array of images.
    images = []

    # For each feature do the following.
    for feature in range(0, 10):
        print("Optimizing image for feature no.", feature)

        # Find the image that maximizes the given feature
        # for the network layer identified by conv_id (or None).
        image = optimize_image(conv_id=conv_id,
                               feature=feature,
                               show_progress=False,
                               num_iterations=num_iterations)

        # Squeeze the dim of the array.
        image = image.squeeze()

        # Append to the list of images.
        images.append(image)

    # Convert to numpy-array so we can index all dimensions easily.
    images = np.array(images)

    # Plot the images.
    plot_images10(images=images)

# First Convolutional Layer
optimize_images(conv_id=0)

# Second Convolutional Layer
optimize_images(conv_id=1)

# Final Output Layer
image = optimize_image(conv_id=None, feature=2, num_iterations=10, show_progress=True)
plot_image(image)

optimize_images(conv_id=None)

# Close TensorFlow Session
session.close()
