# Imports
import os
import time
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from matplotlib.image import imread
from datetime import timedelta
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import confusion_matrix

# Functions and classes for loading and using the Inception model.
import inception
from inception import transfer_values_cache

# Load Data
import knifey
from knifey import num_classes

print('TensorFlow version: {}'.format(tf.__version__))


# knifey.data_dir = "../data/knifey-spoony/"
data_dir = knifey.data_dir

knifey.maybe_download_and_extract()
dataset = knifey.load()

# Your Data
# This is the code you would run to load your own image-files.
# It has been commented out so it won't run now.

# from dataset import load_cached
# dataset = load_cached(cache_path='my_dataset_cache.pk', in_dir='my_images/')
# num_classes = dataset.num_classes

# Training and Test-Sets
class_names = dataset.class_names
print('class_names: {}'.format(class_names))

image_paths_train, cls_train, labels_train = dataset.get_training_set()
print('image_paths_train[0]: {}'.format(image_paths_train[0]))

image_paths_test, cls_test, labels_test = dataset.get_test_set()
print('image_paths_test[0]: {}'.format(image_paths_test[0]))

print("Size of:")
print("- Training-set:\t\t{}".format(len(image_paths_train)))
print("- Test-set:\t\t{}".format(len(image_paths_test)))

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
        # There may be less than 9 images, ensure it doen't crach.
        if i < len(images):
            # Plot iamges.
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

# Helper-function for loading images
def load_images(image_paths):
    # Load the iamges from disk.
    images = [imread(path) for path in image_paths]

    # Convert to a numpy array and retrun it.
    return np.asarray(images)

# Plot a few images to see if data is correct
# Load the first images from the test-set.
indexes = np.random.choice(len(image_paths_test), size=9, replace=False)
images = load_images(image_paths=[image_paths_test[idx] for idx in indexes])

# Get the true classes for those images.
cls_true = cls_test[indexes]

# Plot the images and laabels using our helper-function above.
plot_images(images=images, cls_true=cls_true, smooth=True)

# Download the Inception Model
# inception.data_dir = '../inception/'
inception.maybe_download()

# Load the Inception Model
model = inception.Inception()

# Cauculate Transfer-Values
file_path_cache_train = os.path.join(data_dir, 'inception-knifey-train.pkl')
file_path_cache_test = os.path.join(data_dir, 'inception-knifey-test.pkl')

print("Processing Inception transfer-values for training-images ...")
# If transfer-values have already been calculated then reload them,
# otherwise calculate them and savethem to a cache-file.
transfer_values_train = transfer_values_cache(cache_path=file_path_cache_train,
                                              image_paths=image_paths_train,
                                              model=model)

print("Processing Inception transfer-values for test-images ...")
# If transfer-values have already been calculated then reload them,
# otherwise calculate them and save them to a cache-file.
transfer_values_test = transfer_values_cache(cache_path=file_path_cache_test,
                                             image_paths=image_paths_test,
                                             model=model)

print('transfer_values_train.shape: {}'.format(transfer_values_train.shape))
print('transfer_values_test.shape: {}'.format(transfer_values_test.shape))

# Helper-function for plotting transfer-values
def plot_transfer_values(i):
    print("Input image:")

    # Plot the i'th image from the test-set.
    image = imread(image_paths_test[i])
    plt.imshow(image, interpolation='spline16')
    plt.show()

    print("Transfer-values for the iamge using Inception model:")

    # Transform the transfer-values into an image.
    img = transfer_values_test[i]
    img = img.reshape((32, 64))

    # Plot the iamge for the transfer-values.
    plt.imshow(img, interpolation='nearest', cmap='Reds')
    plt.show()

plot_transfer_values(i=100)
plot_transfer_values(i=300)

# Analysis of Transfer-Values using PCA
pca = PCA(n_components=2)

# transfer_values = transfer_values_train[0:3000]
transfer_values = transfer_values_train
print('transfer_values.shape: {}'.format(transfer_values.shape))

# cls = cls_train[0:3000]
cls = cls_train

transfer_values_reduced = pca.fit_transform(transfer_values)
print('transfer_valeus_reduced.shape: {}'.format(transfer_values_reduced.shape))

def plot_scatter(values, cls):
    # Create a color-map with a different color for each class.
    import matplotlib.cm as cm
    cmap = cm.rainbow(np.linspace(0.0, 1.0, num_classes))

    # Create an index with a random permutation to make a better plot.
    idx = np.random.permutation(len(values))

    # Get the color for each sample.
    colors = cmap[cls[idx]]

    # Extract the x- and y-values.
    x = values[idx, 0]
    y = values[idx, 1]

    # Plot it.
    plt.scatter(x, y, color=colors, alpha=0.5)
    plt.show()

plot_scatter(values=transfer_values_reduced, cls=cls)

# Analysis of Transfer-Values using t-SNE
pca = PCA(n_components=50)
transfer_values_50d = pca.fit_transform(transfer_values)

tsne = TSNE(n_components=2)
transfer_values_reduced = tsne.fit_transform(transfer_values_50d)
print('transfer_values_reduced.shape: {}'.format(transfer_values_reduced.shape))

plot_scatter(values=transfer_values_reduced, cls=cls)

# New classifier in TensorFlow
# Placeholder Variables
transfer_len = model.transfer_len

x = tf.placeholder(tf.float32, shape=[None, transfer_len], name='x')
y_true = tf.placeholder(tf.float32, shape=[None, num_classes], name='y_true')
y_true_cls = tf.math.argmax(input=y_true, axis=1)

# Neural network
layer_fc1 = tf.layers.dense(inputs=x, units=1024, activation=tf.nn.relu, name='layer_fc1')
y_pred = tf.layers.dense(inputs=layer_fc1, units=num_classes, activation=None, name='y_pred')
loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=y_pred, labels=y_true)

# Optimization Method
global_step = tf.get_variable(initializer=0, trainable=False, name='global_step')
optimizer = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(loss, global_step)

# Classification Accuracy
y_pred_cls = tf.math.argmax(input=y_pred, axis=1)
correct_prediction = tf.math.equal(y_pred_cls, y_true_cls)
accuracy = tf.math.reduce_mean(tf.cast(correct_prediction, dtype=tf.float32))

# TensorFlow Run
# Create TensorFlow Session
session = tf.Session()

# Initialize Variables
session.run(tf.global_variables_initializer())

# Helper-function to get a random training-batch
train_batch_size = 64

def random_batch():
    # Number of images (transfer-values) in the trainig-set.
    num_images = len(transfer_values_train)

    # Create a random index.
    idx = np.random.choice(num_images,
                           size=train_batch_size,
                           replace=False)

    # Use the random index to select random x and y-values.
    # We use the ransfer-valeus instead of images of x-values
    x_batch =transfer_values_train[idx]
    y_batch = labels_train[idx]

    return x_batch, y_batch

# Helper-function to perform optimization
def optimize(num_iterations):
    # Start-time used for printing time-usage below.
    start_time = time.time()

    for i in range(num_iterations):
        # Get a batch of training examples.
        # x_batch now holds a batch of images (transfer-values) and
        # y_true_batch are the true labels for those images.
        x_batch, y_true_batch = random_batch()

        # Put the batch into a dict wit hteh proper naems
        # for placeholder variabels in the TensorFlow graph.
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
            # Calcualte teh accuracy on the training-batch.
            batch_acc = session.run(accuracy,
                                    feed_dict=feed_dict_train)

            # Print status.
            msg = "Global Step: {0:>6}, Training Batch Accuracy: {1:>6.1%}"
            print(msg.format(i_global, batch_acc))

    # Ending time.
    end_time = time.time()

    # Difference between start and end-times.
    time_dif = end_time - start_time

    # Print the time-usage.
    print("Time usage: " + str(timedelta(seconds=int(round(time_dif)))))

# Helper-Functions for Showing Results
# Helper-function to plot example errors
def plot_example_errors(cls_pred, correct):
    # This function is called from preint_test_accuracy() below.

    # cls_pred is an array of th epredicted class-number for
    # all images in the test-set.

    # correct is a boolean array whether the predicted class
    # is equal to the true class for each iamge in the test-set.

    # Negate the boolean array.
    incorrect = (correct == False)

    # Get the indices for the incorrectly classified iamges.
    idx = np.flatnonzero(incorrect)

    # number of images to select, max 9.
    n = min(len(idx), 9)

    # Randomize and select n indices.
    idx = np.random.choice(idx,
                           size=n,
                           replace=False)

    # Get the predicted classes for those images.
    cls_pred = cls_pred[idx]

    # Get the true classes for those images.
    cls_true = cls_test[idx]

    # Load the corresponding images fro mteh test-set.
    # Note: We cannot do image_paths_test[idx] on lists of strings.
    image_paths = [image_paths_test[i] for i in idx]
    images = load_images(image_paths)

    # Plot the images.
    plot_images(images=images,
                cls_true=cls_true,
                cls_pred=cls_pred)

# Helper-function to plot confusion matrix
def plot_confusion_matrix(cls_pred):
    # This is called from print_test_accuracy() below.

    # cls_pred is an array of predicted class-number for
    # all images in the test-set.

    # Get the confusion matrix using sklearn.
    cm = confusion_matrix(y_true=cls_test,  # True class for test-set.
                          y_pred=cls_pred)  # Predicted class.

    # Print the confusion matrix as text.
    for i in range(num_classes):
        # Append the class-name to each line.
        class_name = "({}) {}".format(i, class_names[i])
        print(cm[i, :], class_name)

    # Print the class-numbers for eacy reference.
    class_numbers = [" ({0})".format(i) for i in range(num_classes)]
    print("".join(class_numbers))

# Helper-functions for calculating classifications
# Split the data-set in batches of this size to limit RAM usage.
batch_size = 256

def predict_cls(transfer_values, labels, cls_true):
    # Number of images.
    num_images = len(transfer_values)

    # Allocate an array for the predicted classes which
    # will be calculated in batches and filled into this array.
    cls_pred = np.zeros(shape=num_images, dtype=np.int)

    # Now calculate teh predicted classes for the batches.
    # We will just iterate through all the batches.
    # There might be a more clever and Pythonic way of doing this.

    # The starting index for the next batch is denoted i.
    i = 0

    while i < num_images:
        # The ending index for the next batch is denoted j.
        j = min(i + batch_size, num_images)

        # Create a feed-dict with the images and labels
        # between index i and j.
        feed_dict = {x: transfer_values[i:j],
                     y_true: labels[i:j]}

        # Calculate the predicted class using TensorFlow.
        cls_pred[i:j] = session.run(y_pred_cls, feed_dict=feed_dict)

        # Set the start-index for the next batch to the
        # end-index of the current batch.
        i = j

    # Create a boolean array whether each image is correctly classified.
    correct = (cls_true == cls_pred)

    return correct, cls_pred

def predict_cls_test():
    return predict_cls(transfer_values=transfer_values_test,
                       labels=labels_test,
                       cls_true=cls_test)

# Helper-functions for calculating the classification accuracy
def classification_accuracy(correct):
    # When averaging a boolean array, False means 0 and True mean 1.
    # So we are calculating: number of True / len(correct) which is
    # the same as the classification accuracy.

    # Return the classification accuracy
    # and the number of correct classifications.
    return correct.mean(), correct.sum()

# Helper-function for showing the classification accuracy
def print_test_accuracy(show_example_erros=False,
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
    if show_example_erros:
        print("Example errors:")
        plot_example_errors(cls_pred=cls_pred, correct=correct)

    # Plot the confusion matrix, if desired.
    if show_confusion_matrix:
        print("Confusion Matrix:")
        plot_confusion_matrix(cls_pred=cls_pred)

# Results
# Performance before any optimization
print_test_accuracy(show_example_erros=False,
                    show_confusion_matrix=True)

# Performance after 1000 optimization iterations
optimize(num_iterations=1000)
print_test_accuracy(show_example_erros=True,
                    show_confusion_matrix=True)

# Close TensorFlow Session
model.close()
session.close()
