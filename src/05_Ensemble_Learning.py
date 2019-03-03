import os
import math
import time
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.metrics import confusion_matrix
from datetime import timedelta
from mnist import MNIST

# Use PrettyTensor to simplify Neural Network construction.
# import prettytensor as pt

print("Tensorflow version: {0:6}".format(tf.__version__))

# Configuration of Neural Network
# Convolutional layer 1.
filter_size1 = 5        # Convolution filters are 5 x 5 pixels.
num_filters1 = 16       # There are 16 of these filters.

# Convolutional layer 2.
filter_size2 = 5        # Convolution filters are 5 x 5 pixels.
num_filters2 = 36        # There are 36 of these filters.

# Fully-connected layer.
fc_size = 128           # Number of neurons in fully-connected layer.

# Load Data
data = MNIST(data_dir="../data/MNIST")

print("Size of :")
print("- Training-set:\t\t{}".format(data.num_train))
print("- Validation-set:\t{}".format(data.num_val))
print("- Test-set:\t\t{}".format(data.num_test))

# Class numbers
# data.test.cls = np.argmax(data.test.labels, axis=1)
# data.validation.cls = np.argmax(data.validation.labels, axis=1)

# Helper-function for creating random training-sets
combined_images = np.concatenate([data.x_train, data.x_val], axis=0)
combined_labels = np.concatenate([data.y_train, data.y_val], axis=0)

print("combined_images shape: {0}".format(combined_images.shape))
print("combined_labels shape: {0}".format(combined_labels.shape))
print("combined_images dtype: {0}".format(combined_images.dtype))
print("combined_labels dtype: {0}".format(combined_labels.dtype))

combined_size = len(combined_images)
print("combined_size: {0:6}".format(combined_size))

train_size = int(0.8 * combined_size)
print("train_size: {0}".format(train_size))

validation_size = combined_size - train_size
print("validation_size: {}".format(validation_size))

def random_training_set():
    # Create a randomized index into the full / combined training-set.
    idx = np.random.permutation(combined_size)

    # Split the random index into trainining- and validation-sets.
    idx_train = idx[0:train_size]
    idx_validation = idx[train_size:]

    # Select the images and labels for the new training-set.
    x_train_ = combined_images[idx_train, :]
    y_train_ = combined_labels[idx_train, :]

    # Select the images and labels for the new validation-set.
    x_validation = combined_images[idx_validation, :]
    y_validation = combined_labels[idx_validation, :]

    # Return the new training- and validation-sets.
    return x_train_, y_train_, x_validation, y_validation

# Data Dimensions
# We know that MNIST images are 28 pixels in each dimension.
img_size = 28

# Image are stored in one-dimensional arrays of this length.
img_size_flat = img_size * img_size

# Tuple with height and width of iamges used to reshape arrays.
img_shape = (img_size, img_size)

# Number of colour channels for the images: 1 channel for gray-scale.
num_channels = 1

# Number of classes, one class for each of 10 digits.
num_classes = 10

# Helper-function for plotting images
def plot_images(images_,                     # Image to plot, 2-d array.
                cls_true_,                   # True class-no for images.
                ensemble_cls_pred=None,     # Ensemble predicted class-no.
                best_cls_pred=None):        # Best-net predicted class-no.

    assert len(images_) == len(cls_true_)

    # Create figure with 3x3 sub-plots.
    fig, axes = plt.subplots(3, 3)

    # Adjust vertical spacing if we need to pring ensemble and best-net.
    if ensemble_cls_pred is None:
        hspace = 0.3
    else:
        hspace = 1.0
    fig.subplots_adjust(hspace=hspace, wspace=0.3)

    # For each of the sub-plots.
    for idx, ax in enumerate(axes.flat):
        # There may not be enoguh images for all sub-plots.
        if idx < len(images_):
            # Plot image.
            ax.imshow(images_[idx].reshape(img_shape), cmap='binary')

            # Show true and predicted classes.
            if ensemble_cls_pred is None:
                xlabel = "True: {0}".format(cls_true_[idx])
            else:
                msg = "True: {0}\nEnsemble: {1}\nBest Net: {2}"
                xlabel = msg.format(cls_true_[idx],
                                    ensemble_cls_pred[idx],
                                    best_cls_pred[idx])

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
images = data.x_test[0:9]

# Get the true classes for those images.
cls_true = data.y_test_cls[0:9]

# Plot the images and labels using our helper-function above.
plot_images(images_=images, cls_true_=cls_true)

# TensorFlow Graph
x = tf.placeholder(tf.float32, shape=[None, img_size_flat], name='x')
x_image = tf.reshape(x, [-1, img_size, img_size, num_channels])
y_true = tf.placeholder(tf.float32, shape=[None, num_classes], name='y_true')
y_true_cls = tf.argmax(y_true, dimension=1)

# Neural Network
# Helper-functions for creating new variables
def new_weights(shape):
    return tf.Variable(tf.truncated_normal(shape, stddev=0.05))

def new_biases(length):
    return tf.Variable(tf.constant(0.05, shape=[length]))

# Helper-function for creating a new Convolutional Layer
def new_conv_layer(inputs,                  # The previous layer.
                   num_input_channels,      # Num. channels in prev. layer.
                   filter_size,             # Width and height of each filter.
                   num_filters,             # Number of filters.
                   use_pooling=True):       # Use 2x2 max-pooling.

    # Shape of the filter-weights for the convolution.
    # This format is determined by the TensorFlow API.
    shape = [filter_size, filter_size, num_input_channels, num_filters]

    # Create new weights aka. filters with the given shape.
    weights = new_weights(shape=shape)

    # Create new biases, one for each filter.
    biases = new_biases(length=num_filters)

    # Create the TensorFlow operation for convolution.
    # Note teh strides ar eset to 1 in all dimensions.
    # The first and last stride must always be 1,
    # because the first is for the image-number and
    # the last is for the input-channel.
    # But e.g. strides=[1, 2, 2, 1] would mean that the filter
    # is moved 2 pixels across the x- and y-axis fo the image.
    # The padding is set to 'SAME' which means the input iamge
    # is padded with zeroes so the size of the output is the same.
    layer = tf.nn.conv2d(input=inputs,
                         filter=weights,
                         strides=[1, 1, 1, 1],
                         padding='SAME')

    # Add the biases to the results ofthe convolution.
    # A bias-value is added to each filter-channel.
    layer += biases

    # Use pooling to down-sample the iamge resolution?
    if use_pooling:
        # This is 2x2 max-poooling, which means that we
        # conside 2x2 windows and select the largest value
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
    # Layer_shape == [num_images, img_height, img_width, num_channels]

    # The number of features is img_height * img_width * num_channels
    # We can use a function from TensorFlow to calculate this.
    num_features_ = layer_shape[1:4].num_elements()

    # Reshape the layer to [num_images, num_features].
    # Note that we just set the size of the second dimension
    # to num_features and the size of the first dimension to -1
    # which means the size in that dimension is calculated
    # so the total size of the tensor is unchanged from the reshaping.
    layer_flat_ = tf.reshape(layer, [-1, num_features_])

    # The shape of the flattened layer is now:
    # [num_images, img_height * img_width * num_channels]

    # Return both the flattened layer and the number of features.
    return layer_flat_, num_features_

def new_fc_layer(inputs,                        # The previous layer.
                 num_inputs,                    # Num. inputs from prev. layer.
                 num_outputs,                   # Num. outputs.
                 use_relu=True):                # Use Rectified Linear Unit (ReLU)?
    # Create new weights and biases.
    weights = new_weights(shape=[num_inputs, num_outputs])
    biases = new_biases(length=num_outputs)

    # Calculate the layer as the matrix multiplication of
    # the input and weights, and the add the bias-values.
    layer = tf.matmul(inputs, weights) + biases

    # Use ReLU?
    if use_relu:
        layer = tf.nn.relu(layer)

    return layer

# Convolutional Layer 1
layer_conv1, weights_conv1 = new_conv_layer(inputs=x_image,
                                            num_input_channels=num_channels,
                                            filter_size=filter_size1,
                                            num_filters=num_filters1,
                                            use_pooling=True)
print("layer_conv1 shape: {}".format(layer_conv1.get_shape().as_list()))

# Convolutional Layer 2
layer_conv2, weights_conv2 = new_conv_layer(inputs=layer_conv1,
                                            num_input_channels=num_filters1,
                                            filter_size=filter_size2,
                                            num_filters=num_filters2,
                                            use_pooling=True)
print("layer_conv2 shape: {}".format(layer_conv2.get_shape().as_list()))

# Flatten Layer
layer_flat, num_features = flatten_layer(layer_conv2)
print("layer_flat shape: {}".format(layer_flat.get_shape().as_list()))

# Fully-Connected Layer 1
layer_fc1 = new_fc_layer(inputs=layer_flat,
                         num_inputs=num_features,
                         num_outputs=fc_size,
                         use_relu=True)
print("layer_fc1 shape: {}".format(layer_fc1.get_shape().as_list()))

# Fully-Connected Layer 2
layer_fc2 = new_fc_layer(inputs=layer_fc1,
                         num_inputs=fc_size,
                         num_outputs=num_classes,
                         use_relu=False)
print("layer_fc2 shape: {}".format(layer_fc2.get_shape().as_list()))

# Predicted Class
y_pred = tf.nn.softmax(layer_fc2)
y_pred_cls = tf.argmax(y_pred, axis=1)

# Cost-function to be optimized
cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(logits=layer_fc2,
                                                           labels=y_true)
loss = tf.reduce_mean(cross_entropy)

# Optimization Method
optimizer = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(loss)

# Performance measures
correct_prediction = tf.equal(y_pred_cls, y_true_cls)
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# Saver
saver = tf.train.Saver(max_to_keep=100)
save_dir = '../checkpoints/'

if not os.path.exists(save_dir):
    os.makedirs(save_dir)

def get_save_path(net_number):
    return save_dir + 'network' + str(net_number)

# TensorFlow Run
# Create TensorFlow session
session = tf.Session()

# Initialize variables
def init_variables():
    session.run(tf.global_variables_initializer())

# Helper-function to create a random training batch.
train_batch_size = 64

def random_batch(x_train_, y_train_):
    # Total number of images in the trainig-set.
    num_images = len(x_train)

    # Create a random index into the training-set,
    idx = np.random.choice(num_images, size=train_batch_size, replace=False)

    # Use the random index to select reandom images and labels.
    x_batch = x_train_[idx, :]   # Images.
    y_batch = y_train_[idx, :]   # Labels.

    # Return the batch.
    return x_batch, y_batch

def optimize(num_iterations_, x_train_, y_train_):
    # Start-time used for printing time-usage below.
    start_time = time.time()

    for iter_ in range(num_iterations_):
        # Get a batch of training examples.
        # x_batch now holds a batch of images and
        # y_true_batch are the true labels for those iamges.
        x_batch, y_true_batch = random_batch(x_train_, y_train_)

        # Put the batch into a dict with the proper names
        # for placeholder variables in the TensorFlow graph.
        feed_dict_train = {x: x_batch,
                           y_true: y_true_batch}

        # Run the optimize usingthis batch of trainign data.
        # TensorFlow assigns the variables in feed_dict_trian
        # to the placeholder variables and then runs the optimizer.
        session.run(optimizer, feed_dict=feed_dict_train)

        # Print status every 100 iterations and after last iterations.
        if iter_ % 100 == 0:
            # Calculate the accuracy on the training-batch.
            acc = session.run(accuracy, feed_dict=feed_dict_train)

            # Status-message for printing.
            msg = "Optimization Iteration: {0:>6}, Training Batch Accuracy: {1:>6.1%}"

            # Print it.
            print(msg.format(iter_+1, acc))

    # Ending time.
    end_time = time.time()

    # Difference between start and end-times.
    time_dif = end_time - start_time

    # Print the time-usage.
    print("Time usage: " + str(timedelta(seconds=int(round(time_dif)))))

# Create ensemble of neural networks.
num_networks = 5
num_iterations = 10000

is_ensemble = True
if is_ensemble:
    # For each of the neural networks.
    for i in range(num_networks):
        print("Neural network: {0}".format(i))

        # Create a random training-set. Ignore the validation-set.
        x_train, y_train, _, _ = random_training_set()

        # Initialize the variables of the TensorFlow graph.
        session.run(tf.global_variables_initializer())

        # Optimize the variables using this training-set.
        optimize(num_iterations_=num_iterations,
                 x_train_=x_train,
                 y_train_=y_train)

        # Save the optimized variables to disk.
        saver.save(sess=session, save_path=get_save_path(i))

        # Print newline.
        print()

# Helper-functions for calculating and predicting classifications
# Split the data-set in batchs of this size to limit RAM usage.
batch_size = 256

def predict_labels(images_):
    # Number of images.
    num_images = len(images_)

    # Allocate an array for the predicted labels which
    # will be calculated in batches and filled into this array.
    pred_labels_ = np.zeros(shape=(num_images, num_classes), dtype=np.float32)

    # Now calculate the predicted labels for the batches.
    # We will just iterate through all the batches.
    # There might be a more clever and Pythonic way of doing this.

    # The starting index ofr the next batch is denoted i.
    idx_i = 0

    while idx_i < num_images:
        # The ending index for the next batch is denoted j.
        idx_j = min(idx_i + batch_size, num_images)

        # Create a feed-dict with the images between index i and j.
        feed_dict = {x: images_[idx_i:idx_j, :]}

        # Calculate the predicted labels using TensorFlow.
        pred_labels_[idx_i:idx_j] = session.run(y_pred, feed_dict=feed_dict)

        # Set the start-index for the next batch to the
        # end-index of the current batch.
        idx_i = idx_j

    return pred_labels_

def correct_prediction(images_, labels_, cls_true_):
    # Calculate the predicted labels.
    pred_labels_ = predict_labels(images_=images_)

    # Calculate the predicted class-number for each image.
    cls_pred = np.argmax(pred_labels_, axis=1)

    # Create a boolean array whether each image is correctly classified.
    correct = (cls_true_ == cls_pred)

    return correct

def test_correct():
    return correct_prediction(images_=data.x_test,
                              labels_=data.y_test,
                              cls_true_=data.y_test_cls)

def validation_correct():
    return correct_prediction(images_=data.x_val,
                              labels_=data.y_val,
                              cls_true_=data.y_val_cls)

# Helper-functions for calculating the classification accuracy
def classification_accuracy(correct):
    # When averaging a boolean array, False means 0 and True means 1.
    # So we are calculating: number of True / len(correct) which is
    # the same as the classification accuracy.
    return correct.mean()

def test_accuracy():
    # Get the array of booleans whether the classifications are correct
    # for the test-set.
    correct = test_correct()

    # Calculate the classification accuracy and return it
    return classification_accuracy(correct)

def validation_accuracy():
    # Get the array of booleans whether the classificatiosn are correct
    # for the validation-set.
    correct = validation_correct()

    # Calculate the classification accuracy and return it.
    return classification_accuracy(correct)

# Results and analysis
def ensemble_predictions():
    # Empty list of predicted labels for each of the neural networks.
    pred_labels_ = []

    # Classification accuracy on the test-set for each network.
    test_accuracies_ = []

    # Classification accuracy on the validation-set for each network.
    val_accuracies_ = []

    # For each neural network in the ensemble.
    for idx in range(num_networks):
        # Reload the variables into the TensorFlow graph.
        saver.restore(sess=session, save_path=get_save_path(idx))

        # Calculate the classification accuracy on the test-set.
        test_acc = test_accuracy()

        # Append the classification accuracy to the list.
        test_accuracies_.append(test_acc)

        # Calculate the classification accuracy on the validation-set.
        val_acc = validation_accuracy()

        # Append the classification accuracy to the list.
        val_accuracies_.append(val_acc)

        # Print status message.
        msg = "Network: {0}, Accuracy on Validation-Set: {1:.4f}, Test-Set: {2:.4f}"
        print(msg.format(idx, val_acc, test_acc))

        # Calculate the predicted labels for the images in the test-set.
        # This is already calculated in test_accuracy() above but
        # it is re-calculated here to keep the code a bit simpler.
        pred = predict_labels(images_=data.x_test)

        # Appende the predicted labels to the list.
        pred_labels_.append(pred)

    return np.array(pred_labels_), np.array(test_accuracies_), np.array(val_accuracies_)

pred_labels, test_accuracies, val_accuracies = ensemble_predictions()

print("Mean test-set accuracy: {0:.4f}".format(np.mean(test_accuracies)))
print("Min test-set accuracy: {0:.4f}".format(np.min(test_accuracies)))
print("Max test-set accuracy: {0:.4f}".format(np.max(test_accuracies)))

# Ensemble predictions
ensemble_pred_labels = np.mean(pred_labels, axis=0)
print('ensemble_pred_labels shape: {}'.format(ensemble_pred_labels.shape))

ensemble_cls_pred = np.argmax(ensemble_pred_labels, axis=1)
print('ensemble_cls_pred shape: {}'.format(ensemble_cls_pred.shape))

ensemble_correct = (ensemble_cls_pred == data.y_test_cls)
ensemble_incorrect = np.logical_not(ensemble_correct)

# Best neural network
print('test accuracies: {}'.format(test_accuracies))

best_net = np.argmax(test_accuracies)
print('best_net: {}'.format(best_net))

print('test_accuracies[best_net]: {}'.format(test_accuracies[best_net]))

best_net_pred_labels = pred_labels[best_net, :, :]
best_net_cls_pred = np.argmax(best_net_pred_labels, axis=1)

best_net_correct = (best_net_cls_pred == data.y_test_cls)
best_net_incorrect = np.logical_not(best_net_correct)

# Comparison of ensemble vs. the best single network
print('np.sum(ensemble_correct): {}'.format(np.sum(ensemble_correct)))
print('np.sum(best_net_correct): {}'.format(np.sum(best_net_correct)))

ensemble_better = np.logical_and(best_net_incorrect, ensemble_correct)
print('ensemble_better.sum(): {}'.format(ensemble_better.sum()))

best_net_better = np.logical_and(best_net_correct, ensemble_incorrect)
print('best_net_better.sum(): {}'.format(best_net_better.sum()))

# Helper-functions for plotting and printing comparsions
def plot_images_comparison(idx):
    plot_images(images_=data.x_test[idx],
                cls_true_=data.y_test_cls[idx],
                ensemble_cls_pred=ensemble_cls_pred[idx],
                best_cls_pred=best_net_cls_pred[idx])

def print_labels(labels, idx, num=1):
    # Select the relevant labels based on idx.
    labels =labels[idx, :]

    # Select the first num labels.
    labels = labels[0:num, :]

    # Round numbers to 2 decimal points so they are easier to read.
    labels_rounded = np.round(labels, 2)

    # Print the rounded labels.
    print(labels_rounded)

def print_labels_ensemble(idx, **kwargs):
    print_labels(labels=ensemble_pred_labels, idx=idx, **kwargs)

def print_labels_best_net(idx, **kwargs):
    print_labels(labels=best_net_pred_labels, idx=idx, **kwargs)

def print_labels_all_nets(idx):
    for i in range(num_networks):
        print_labels(labels=pred_labels[i, :, :], idx=idx, num=1)

    # Examples: Ensemble is better than the best network
    plot_images_comparison(idx=ensemble_better)

    print_labels_ensemble(idx=ensemble_better, num=1)

    print_labels_best_net(idx=ensemble_better, num=1)

    print_labels_all_nets(idx=ensemble_better)

    # Examles: Best network is bettern than ensemble
    plot_images_comparison(idx=best_net_better)

    print_labels_ensemble(idx=best_net_better, num=1)

    print_labels_best_net(idx=best_net_better, num=1)

    print_labels_all_nets(idx=best_net_better)

    # Close TensorFlow Session
    # This had been cmmented out in case you want to modify and experiment
    # with the Notebook without having to restart it.
    session.close()











