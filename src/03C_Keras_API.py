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
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.models import load_model
# from tensorflow.python.keras.layers import Adam
from tensorflow.python.keras import backend as K

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
def plot_images(images_, cls_true_, cls_pred_=None):
    assert len(images_) == len(cls_true_) == 9

    # Create figure with 3x3 sub-plots.
    fig, axes = plt.subplots(3, 3)
    fig.subplots_adjust(hspace=0.3, wspace=0.3)

    for i, ax in enumerate(axes.flat):
        # Plot image.
        ax.imshow(images_[i].reshape(img_shape), cmap='binary')

        # Show true and predicted classes.
        if cls_pred_ is None:
            xlabel = "True: {0}".format(cls_true_[i])
        else:
            xlabel = "True: {0}, Pred: {1}".format(cls_true_[i], cls_pred_[i])

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
def plot_example_errors(cls_pred_):
    # cls_pred is an array of the predicted class-number for
    # all images in the test-set.

    # Boolean array whether the predicted class is incorrect.
    incorrect = (cls_pred_ != data.y_test_cls)

    # Get the iamges from the test-set that have been
    # incorrrectly classified.
    images_ = data.x_test[incorrect]

    # Get the predicted classes for those images.
    cls_pred_ = cls_pred_[incorrect]

    # Get the true classes for those images.
    cls_true_ = data.y_test_cls[incorrect]

    # Plot the first 9 images.
    plot_images(images_=images_[0:9],
                cls_true_=cls_true_[0:9],
                cls_pred_=cls_pred_[0:9])

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

print('Model-1 Test!')
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

# Prediciton
images = data.x_test[0:9]
cls_true = data.y_test_cls[0:9]
y_pred = model.predict(x=images)
cls_pred = np.argmax(y_pred, axis=1)
plot_images(images_=images,
            cls_true_=cls_true,
            cls_pred_=cls_pred)

# Examples of Mis-Classified Images
y_pred = model.predict(x=data.x_test)
cls_pred = np.argmax(y_pred, axis=1)
plot_example_errors(cls_pred)

print('Model-2 Test!')
# Functional Model
# Create an input layer which is similar to a feed_dict in TensorFlow.
# Note that the input-shape must be a tulpe containing the image-size.
inputs = Input(shape=(img_size_flat,))

# Variable used for building theNeural Network.
net = inputs

# The input is an image as a flattened array with 784 elements.
# But the convolutional layers expect image with shape (28, 28, 1)
net = Reshape(img_shape_full)(net)

# First convolutional layer with ReLU-activation and max-pooling.
net = Conv2D(kernel_size=5, strides=1, filters=16, padding='same', activation='relu', name='layer_conv1')(net)
net = MaxPooling2D(pool_size=2, strides=2)(net)

# Second convolutional layer with ReLU-activation and max-pooling.
net = Conv2D(kernel_size=5, strides=1, filters=36, padding='same', activation='relu', name='layer_conv2')(net)
net = MaxPooling2D(pool_size=2, strides=2)(net)

# Flatten the output of the conv-layer from 4-dim to 2-dim.
net = Flatten()(net)

# Last fully-connect / dense layer with ReLU-activation.
net = Dense(128, activation='relu')(net)

# Last fully-connect / dense layer with softmax-activation
# so it can be used for classification.
net = Dense(num_classes, activation='softmax')(net)

# Output of the Neural Network.
outputs = net

# Model Compilation
model2 = Model(inputs=inputs, outputs=outputs)

model2.compile(optimizer='rmsprop',
               loss='categorical_crossentropy',
               metrics=['accuracy'])

# Training
model2.fit(x=data.x_train,
           y=data.y_train,
           epochs=1, batch_size=128)

# Evaluation
result = model2.evaluate(x=data.x_test,
                         y=data.y_test)

for name, value in zip(model2.metrics_names, result):
    print(name, value)

print("{0}: {1:.2%}".format(model2.metrics_names[1], result[1]))

# Examples of Mis-Classified Images
y_pred = model2.predict(x=data.x_test)
cls_pred = np.argmax(y_pred, axis=1)
plot_example_errors(cls_pred)

# Save & Load Model
path_model = '../checkpoints/model.keras'
model2.save(path_model)
print("Success to write!")
del model2

print("Model-3 Test!")
model3 = load_model(path_model)
print("Success to read!")
images = data.x_test[0:9]
cls_true = data.y_test_cls[0:9]
y_pred = model3.predict(x=images)
cls_pred = np.argmax(y_pred, axis=1)
plot_images(images_=images,
            cls_pred_=cls_pred,
            cls_true_=cls_true)

# Visualization of Layer Weights and Outputs
# Helper-function for plotting convolutional weights
def plot_conv_weights(weights, input_channel=0):
    # Get the lowest and highest values for the weights.
    # This is used to correct the colour intensity across
    # the images so they can be compared with each oterh.
    w_min = np.min(weights)
    w_max = np.max(weights)

    # Number of filters used in the conv. layer.
    num_filters = weights.shape[3]

    # Number of grids to plot.
    # Rounded-up, square-root of the number of filters.
    num_grids = math.ceil(math.sqrt(num_filters))

    # Create figure with a grid of sub-plots.
    fig, axes = plt.subplots(num_grids, num_grids)

    # Plot all filter-weights.
    for i, ax in enumerate(axes.flat):
        # Only plot the valid filter-weights.
        if i < num_filters:
            # Get the weights for i'th filter of the input channel.
            # See new_conv_layer() for details on the format
            # of this 4-dim tensor.
            img = weights[:, :, input_channel, i]

            # Plot image.
            ax.imshow(img, vmin=w_min, vmax=w_max, interpolation='nearest', cmap='seismic')

        # Remove ticks from the plot.
        ax.set_xticks([])
        ax.set_yticks([])

    # Ensure the plot is shown correctly with multiple plots
    # in a single Notebook cell.
    plt.show()

# Get layers
model3.summary()
layer_input = model3.layers[0]
layer_conv1 = model3.layers[2]
layer_conv2 = model3.layers[4]

# Convolutional Weights
weights_conv1 = layer_conv1.get_weights()[0]
print('weights_conv1 shape: {}'.format(weights_conv1.shape))
plot_conv_weights(weights=weights_conv1, input_channel=0)

weights_conv2 = layer_conv2.get_weights()[0]
plot_conv_weights(weights=weights_conv2, input_channel=0)

# Helper-function for plotting the output of a convolutional layer
def plot_conv_output(values):
    # Number of filters used in the conv. layer.
    num_filters = values.shape[3]

    # Number of grids to plot.
    # Rounded-up, square-root of the number of filters.
    num_grids = math.ceil(math.sqrt(num_filters))

    # Creat efigure with a grid of sub-plots.
    fig, axes = plt.subplots(num_grids, num_grids)

    # Plot the output iamges of all the filters.
    for i, ax in enumerate(axes.flat):
        # Only plot the iamges for valid filters.
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

# Input Image
def plot_image(image):
    plt.imshow(image.reshape(img_shape),
               interpolation='nearest',
               cmap='binary')
    plt.show()

image1 = data.x_test[0]
plot_image(image1)

# Output of Convolutional Layer - Method 1
output_conv1 = K.function(inputs=[layer_input.input],
                          outputs=[layer_conv1.output])

layer_output1 = output_conv1([[image1]])[0]
print('layer_output1.shape: {}'.format(layer_output1.shape))

plot_conv_output(values=layer_output1)

# Output of Convolutional Layer - Method 2
output_conv2 = Model(inputs=layer_input.input,
                     outputs=layer_conv2.output)

layer_output2 = output_conv2.predict(np.array([image1]))
print('layer_output2.shape: {}'.format(layer_output2.shape))

plot_conv_output(values=layer_output2)
