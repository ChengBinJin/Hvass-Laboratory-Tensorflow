# Imports
import math
import time
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from datetime import timedelta
from sklearn.metrics import confusion_matrix

print('TensorFlow version: {0}'.format(tf.__version__))

# Load Data
from mnist import MNIST
data = MNIST(data_dir="../data/MNIST/")

print("Size of:")
print("- Training-set:\t\t{}".format(data.num_train))
print("- Validation-set:\t{}".format(data.num_val))
print("- Test-set:\t\t{}".format(data.num_test))

# The number of pixels in each dimension of an image.
img_size = data.img_size
print("- img_size:\t\t{}".format(img_size))

# The images are stored in one-dimensional arrays of this length.
img_size_flat = data.img_size_flat
print("- img_size_flat:\t{}".format(img_size_flat))

# Tuple with height and width of images used to reshape arrays.
img_shape = data.img_shape
print("- img_shape:\t\t{}".format(img_shape))

# Number of classes, one class for each 10 digits.
num_classes = data.num_classes
print("- num_classes:\t{}".format(num_classes))

# Number of colour channels for the images: 1 channel for gray-scale.
num_channels = data.num_channels
print("- num_channels:\t{}".format(num_channels))


# Helper-function for plotting images
def plot_images(images, cls_true, cls_pred=None, noise=0.0):
    assert len(images) == len(cls_true) == 9

    # Create figure with 3x3 sub-plots.
    fig, axes = plt.subplots(3, 3)
    fig.subplots_adjust(hspace=0.3, wspace=0.3)

    for i, ax in enumerate(axes.flat):
        # Get the i'th image and reshape the array.
        image = images[i].reshape(img_shape)

        # Add the adversarial noise to the image.
        image += noise

        # Ensure the noisy pixel-values are between 0 and 1.
        image = np.clip(a=image, a_min=0.0, a_max=1.0)

        # Plot image.
        ax.imshow(image, cmap='binary', interpolation='nearest')

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
# Get the first iamges from the test-set.
idxes = np.random.random_integers(low=0, high=data.num_test, size=9)
images = data.x_test[idxes]

# Get the true classes for those images.
cls_true = data.y_test_cls[idxes]

# Plot the images and labels using our helper-function above.
plot_images(images=images, cls_true=cls_true)

# TensorFlow Graph
# Placeholder variables
x = tf.placeholder(tf.float32, shape=[None, img_size_flat], name='x')
x_image = tf.reshape(x, [-1, img_size, img_size, num_channels])
y_true = tf.placeholder(tf.float32, shape=[None, num_classes], name='y_true')
y_true_cls = tf.math.argmax(input=y_true, axis=1)

# Adversarial Noise
noise_limit = 0.35
noise_l2_weight = 0.02

ADVERSARY_VARIABLES = 'adversary_variables'
collections = [tf.GraphKeys.GLOBAL_VARIABLES, ADVERSARY_VARIABLES]

x_noise = tf.get_variable(name='x_noise',
                          shape=[img_size, img_size, num_channels],
                          initializer=tf.keras.initializers.Zeros(),
                          trainable=False,
                          collections=collections)
x_noise_clip = tf.assign(ref=x_noise,
                         value=tf.clip_by_value(t=x_noise,
                                                clip_value_min=-noise_limit,
                                                clip_value_max=noise_limit))

x_noisy_image = x_image + x_noise
x_noisy_image = tf.clip_by_value(t=x_noisy_image,
                                 clip_value_min=0.0,
                                 clip_value_max=1.0)

# Convolutional Neural Network
# Start the network with the noisy input image.
net = x_noisy_image

# 1st convolutional layer.
net = tf.layers.conv2d(inputs=net,
                       name='layer_conv1',
                       padding='same',
                       filters=16,
                       kernel_size=5,
                       activation=tf.nn.relu)
net = tf.layers.max_pooling2d(inputs=net,
                              pool_size=2,
                              strides=2)

# 2nd convolutional layer.
net = tf.layers.conv2d(inputs=net,
                       name='layer_conv2',
                       padding='same',
                       filters=36, kernel_size=5, activation=tf.nn.relu)
net = tf.layers.max_pooling2d(inputs=net,
                              pool_size=2,
                              strides=2)

# Flatten layer. This should eventually be replaced by:
net = tf.layers.flatten(inputs=net)

# 1st fully-connect /dense layer.
net = tf.layers.dense(inputs=net,
                      name='layer_fc1',
                      units=128,
                      activation=tf.nn.relu)

# 2nd fully-connect / dense layer.
net = tf.layers.dense(inputs=net,
                      name='layer_fc_out',
                      units=num_classes,
                      activation=None)

# Unscaled output of the network.
logits = net

# Softmax output of the network.
y_pred = tf.nn.softmax(logits=logits)

# Loss measure to be optimized.
cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(labels=y_true,
                                                           logits=logits)

loss = tf.reduce_mean(cross_entropy)

# Optimizer for Normal Training
for var in tf.trainable_variables():
    print(var.name)

optimizer = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(loss)

# Optimizer for Adversarial Noise
adversary_variables = tf.get_collection(ADVERSARY_VARIABLES)

for var in adversary_variables:
    print(var.name)

l2_loss_noise = noise_l2_weight * tf.nn.l2_loss(t=x_noise)
loss_adversary = loss + l2_loss_noise

optimizer_adversary = tf.train.AdamOptimizer(learning_rate=1e-2).minimize(loss_adversary, var_list=adversary_variables)

# Performance Measure
y_pred_cls = tf.math.argmax(input=y_pred, axis=1)
correct_prediction = tf.equal(y_pred_cls, y_true_cls)
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# TensorFlow Run
# Create TensorFlow session
session = tf.Session()

# Initialize variables
session.run(tf.global_variables_initializer())

def init_noise():
    session.run(tf.variables_initializer([x_noise]))

init_noise()

# Helper-function to perform optimization iterations
train_batch_size = 64


def optimize(num_iterations, adversary_target_cls=None):
    # Start-time used for printing time-usage below.
    start_time = time.time()

    for i in range(num_iterations):
        # Get a batch of training examples.
        # x_batch now holds a batch of iamges and
        # y_true_batch are the true labels for those iamges.
        x_batch, y_true_batch, _ = data.random_batch(batch_size=train_batch_size)

        # If we are searching for the adversarial noise, then
        # use the adversarial target-class instead.
        if adversary_target_cls is not None:
            # The class-labels are One-Hot encoded.

            # Set all the class-labels to zero.
            y_true_batch = np.zeros_like(y_true_batch)

            # Set the element for the adversarial target-class to 1.
            y_true_batch[:, adversary_target_cls] = 1.0

        # Put hte batch into a dict with the proper names
        # for placeholder variables in the TensorFlow graph.
        feed_dict_train = {x: x_batch,
                           y_true: y_true_batch}

        # If doing normal optimization of the neural network.
        if adversary_target_cls is None:
            # Run the optimizer using this batch of training data.
            # TensorFlow assigns the variables in feed_dict_train
            # to the placeholder variables and then runs the optimzier.
            session.run(optimizer, feed_dict=feed_dict_train)
        else:
            # Run the adversarial optimizer instead.
            # Note that we have 'faked' the class above to be
            # the adversarial target-calss instead of the true class.
            session.run(optimizer_adversary, feed_dict=feed_dict_train)

            # Clip /; limit the adversarial noise. This executes
            # another TensorFlow operation. It cannot be executed
            # in the same session.run() as the optimizer, because
            # it may run in parallel so the execution order is not
            # guaranteed. We need the clip to run after the optimizer.
            session.run(x_noise_clip)

        # Print status every 100 iterations.
        if (i % 100 == 0)  or (i == num_iterations - 1):
            # Calculate the accuracy on the training-set.
            acc, data_loss = session.run([accuracy, loss], feed_dict=feed_dict_train)

            # Message for printing.
            msg = "Optimization Iteration: {0:>6}, Loss: {1:>10.5}, Training Accuracy: {2:>6.2%}"

            # Print it.
            print(msg.format(i, data_loss, acc))

    # Ending time.
    end_time = time.time()

    # Difference between start and end-times.
    time_dif = end_time - start_time

    # Print the time-usage.
    print("Time usage: " + str(timedelta(seconds=int(round(time_dif)))))


# Helper-functions for getting and plotting the noise
def get_noise():
    # Run the TensorFlow session to retrieve the contents of
    # the x_noise variable inside the graph.
    noise = session.run(x_noise)

    return noise.squeeze()


def plot_noise():
    # Get the adversarial noise from inside the TensorFlow graph.
    noise = get_noise()

    # Print statistics.
    print("Noise:")
    print("- Min: {:>6.3}".format(noise.min()))
    print("- Max: {:>6.3}".format(noise.max()))
    print("- Std: {:>6.3}".format(noise.std()))

    # Plot the noise.
    plt.imshow(noise, interpolation='nearest', cmap='seismic', vmin=-1.0, vmax=1.0)

# Helper-function to plot example errors
def plot_example_errors(cls_pred, correct):
    # This function is called from print_test_accuracy() below.

    # clas_pred is an array of the predicted class-number for
    # all images in the test-set.

    # correct is a boolean array whether the predicted class
    # is equal to the true class for each image in the test-set.

    # Negate the boolean array.
    incorrect = (correct == False)

    # Get the images from the test-set that have been
    # incorrectly classified.
    images = data.x_test[incorrect]

    # Get the predicted classes for those images.
    cls_pred = cls_pred[incorrect]

    # Get the true classes for those images.
    cls_true = data.y_test_cls[incorrect]

    # Get the adversarial noise from inside the TensorFlow graph.
    noise = get_noise()

    # Plot the first 9 images.
    plot_images(images=images[:9],
                cls_true=cls_true[:9],
                cls_pred=cls_pred[:9],
                noise=noise)


# Helper-function to plot confusion matrix
def plot_confusion_matrix(cls_pred):
    # This is called from print_test_accuracy() below.

    # cls_pred is an array of the predicted class-number for
    # all images in the test-set.

    # Get the true classifications for the test-set.
    cls_true =data.y_test_cls

    # Get the confusion matrix using sklearn.
    cm = confusion_matrix(y_true=cls_true,
                          y_pred=cls_pred)

    # Print the confusion matrix as text.
    print(cm)


# Helper-function for showing the performance
# Split the test-set into smaller batches of this size.
test_batch_size = 512

def print_test_accuracy(show_example_errors=False, show_confusion_matrix=False):
    # Number of images in the test-set.
    num_test = data.num_test

    # Allocate an array for the predicted classes which
    # will be calculated in batches and filled into this array.
    cls_pred = np.zeros(shape=num_test, dtype=np.uint8)

    # Now calculate the predicted classes for the batches.
    # We will just iterate through all the batches.
    # There might be a more clever and Pythonic way of doing this.

    # The starting index for the next batch is denoted i.
    i = 0

    while i < num_test:
        #  The ending index for the next batch is denoted j.
        j = min(i + test_batch_size, num_test)

        # Get the iamges from teh test-set between index i and j.
        images = data.x_test[i:j, :]

        # Get the associated labels.
        labels = data.y_test[i:j, :]

        # Create a feed-dict with these iamges and labels.
        feed_dict = {x: images,
                     y_true: labels}

        # Calculate the predicted class using TensorFlow.
        cls_pred[i:j] = session.run(y_pred_cls, feed_dict=feed_dict)

        # Set the start-index for the next batch to the
        # end-index of the current batch.
        i = j

    # Convenience variable for the true class-numbers of the test-set.
    cls_true = data.y_test_cls

    # Create a boolean array whether each iamge is correctly classified.
    correct = (cls_true == cls_pred)

    # Calculate the number of correctly classified iamges.
    # When summing a boolean array, False mean 0 and True means 1.
    correct_sum = correct.sum()

    # Classification accuracy is the number of correctly classified
    # images divided by the total numbe of iamges in the test-set.
    acc = float(correct_sum) / num_test

    # Print the accuracy.
    msg = "Accuracy on Test-Set: {0:.2%} ({1} / {2})"
    print(msg.format(acc, correct_sum, num_test))

    # Plot some examples of mis-classifications, if desired.
    if show_example_errors:
        print("Example errors:")
        plot_example_errors(cls_pred=cls_pred, correct=correct)

    # Plot the confuion matrix, if desire.
    if show_confusion_matrix:
        print("Confusion Matrix:")
        plot_confusion_matrix(cls_pred=cls_pred)


# Normal optimization of neural network
optimize(num_iterations=1000)
print_test_accuracy(show_example_errors=True)

# Find the adversarial noise
init_noise()

optimize(num_iterations=1000, adversary_target_cls=3)

plot_noise()

print_test_accuracy(show_example_errors=True,
                    show_confusion_matrix=True)

# Adversarial noise for all target-classes
def find_all_noise(num_iterations=1000):
    # Adversarial noise for all target-classes.
    all_noise = []

    # For each target-class.
    for i in range(num_classes):
        print("Finding adversarial noise for target-class:", i)

        # Reset the adversarial noise to zero.
        init_noise()

        # Optimize the adversarial noise.
        optimize(num_iterations=num_iterations,
                 adversary_target_cls=i)

        # Get the adversarial noise from inside the TensorFlow graph.
        noise = get_noise()

        # Append the noise to the array.
        all_noise.append(noise)

        # Print newline.
        print()

    return all_noise

all_noise = find_all_noise(num_iterations=300)

# Plot the adversarial noise for all target-classes
def plot_all_noise(all_noise):
    # Create figure with 10 sub-plots.
    fig, axes = plt.subplots(2, 5)
    fig.subplots_adjust(hspace=0.2, wspace=0.1)

    # For each sub-plot.
    for i, ax in enumerate(axes.flat):
        # Get the adversarial noise for the i'th target-class.
        noise = all_noise[i]

        # Plot the noise.
        ax.imshow(noise,
                  cmap='seismic', interpolation='nearest',
                  vmin=-1.0, vmax=1.0)

        # Show the classes as the label on the x-axis.
        ax.set_xlabel(i)

        # Remove ticks from the plot.
        ax.set_xticks([])
        ax.set_yticks([])

    # Ensure the plot is shown correctly with multiple plots
    # in a single Notebook cell.
    plt.show()

plot_all_noise(all_noise)



# Immunity to adversarial noise
# Helper-function to make a neural network immune to noise

# Make immune to noise for target-class 3

# make immune to noise for all target-classes



# Make immune to all target-classes (double runs)

# Plot the adversarial noise

# Performance on clean images



# Close TensorFlow Session
