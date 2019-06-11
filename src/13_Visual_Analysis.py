# TensorFlow Tutorial #13
# Visual Analysis
# Introduction
# Flowchart

# Imports
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

# Functions and classes for loading and using the Inception model.
import inception

print('TensorFlow Version: {}'.format(tf.__version__))

# Inception Model
# Download the Inception model from the interne
inception.data_dir = '../inception'
inception.maybe_download()

# Names of convolutional layers
def get_conv_layer_names():
    # Load the Inception model.
    model = inception.Inception()

    # Create a list of names for the operations in the graph
    # for the Inception model where the operator-type is 'Conv2D'.
    names = [op.name for op in model.graph.get_operations() if op.type == 'Conv2D']

    # Close the TensorFlow session inside the model-object.
    model.close()

    return names

conv_names = get_conv_layer_names()
print('len(conv_names): {}'.format(len(conv_names)))

print('First 5 layers: {}'.format(conv_names[:5]))
print('Last 5 layers: {}'.format(conv_names[-5:]))


# Helper-function for finding the inptu image
def optimize_image(conv_id=None, feature=0, num_iterations=30, show_progress=True):
    # Find and image that maximizes the feature
    # given by the conv_id and feature number.
    # Parameters:
    # conv_id:  Integer identifying the convolutional layer to
    #           maximize. It is an index into conv_names.
    #           If None then the last fully-connected layer
    #           before the softmax output.
    # feature:  Index into the layer for the feature to maximize.
    # num_iteration: Number of optimization iterations to perform.
    # show_progress: Boolean whether to show the progress.

    # Load the Inception model. this is done for each call of
    # this function because we will add a lot ot the graph
    # which will cause the graph to grow and eventually the
    # computer will run out of memory.
    model = inception.Inception()

    # Reference to the tensor that takes the raw input image.
    resized_image = model.resized_image

    # Reference to the tensor for the predicted classes.
    # This is the output of the final layer's softmax classifier.
    y_pred = model.y_pred

    # Create the loss-function that must be maximized.
    if conv_id is None:
        # If we want to maimize a feature on the last layer,
        # then we use the fully-connected layer prior to the
        # softmax-classifier. The feature no. is the class-number
        # and must be an integer between 1 and 1000.
        # the loss-function is just the value of that feature.
        loss = model.y_logits[0, feature]
    else:
        # If instead we want to maximize a feature of a
        # convolutional layer inside the neural network.

        # Get the name of the convolutional operator.
        conv_name = conv_names[conv_id]

        # Get a reference to the tensor that is output by the
        # operator. Note that ":0" is added to the name for this.
        tensor = model.graph.get_tensor_by_name(conv_name + ":0")

        # Set the Inception model's graph as the default
        # so we can add an operator to it.
        with model.graph.as_default():
            # The loss-function is the average of all the
            # tensor-values for the given feature. This
            # ensures that we generate the whole input image.
            # You can try and modify this so it only uses
            # a part of the tensor.
            loss = tf.reduce_mean(tensor[:, :, :, feature])

    # Get the gradient for the loss-function with regard to
    # the resized input image. This creates a mathematical
    # function for calculating the gradient.
    gradient = tf.gradients(loss, resized_image)

    # Create a TensorFlow session so we can run the graph.
    session = tf.Session(graph=model.graph)

    # Generate a random image of the same size as teh raw input.
    # Each pixel is a small random value between 128 and 129,
    # which is about the middle of the colour-range.
    image_shape = resized_image.get_shape()
    image = np.random.uniform(size=image_shape) + 128.0

    # Perform a number of optimization iterations to find
    # the image that maximizes the loss-function.
    for i in range(num_iterations):
        # Create a feed-dict. This feeds the iamge to the
        # tensor in teh graph that holds the resized iamge, because
        # this is the final stage for inputting raw image data.
        feed_dict = {model.tensor_name_resized_image: image}

        # Calculate the predicted class-scores,
        # as well as the gradient and the loss-value.
        pred, grad, loss_value = session.run([y_pred, gradient, loss],
                                             feed_dict=feed_dict)

        # Squeeze the dimensionality for the gradient-array.
        grad = np.array(grad).squeeze()

        # The gradient now tells us how much we need to change the
        # input image in order to maximize the given feature.

        # Calculate the step-size for updating the image.
        # This step-size was found to give fast convergence.
        # The addition of 1e-8 is to protect from div-by-zero.
        step_size = 1.0 / (grad.std() + 1e-8)

        # Update the iamge by adding the scaled gradient
        # This is called gradient ascent.
        image += step_size * grad

        # Ensure all pixel-values in the image are between 0 and 255.
        image = np.clip(image, a_min=0., a_max=255.)

        if show_progress:
            print("Iteration: {}".format(i))

            # Convert the predicted class-score to a one-dim array
            pred = np.squeeze(pred)

            # The predicted class for the Inception model
            pred_cls = np.argmax(pred)

            # Name of the predicted class.
            cls_name = model.name_lookup.cls_to_name(pred_cls, only_first_name=True)

            # The score (probability) for the predicted class.
            cls_score = pred[pred_cls]

            # Print the predicted score etc.
            msg = "Predicted class-name: {0} (#{1}), socre: {2:7.2%}"
            print(msg.format(cls_name, pred_cls, cls_score))

            # Print statistics for the gradient.
            msg = "Gradient min: {0:>9.6f}, max: {1:>9.6f}, stepsize: {2:>9.2f}"
            print(msg.format(grad.min(), grad.max(), step_size))

            # Print the loss-value.
            print("Loss: {}".format(loss_value))

            # Newline.
            print()

    # Close the TensorFlow session inside the model-object.
    model.close()

    return image.squeeze()

# Helper-function for plotting image and noise
def normalize_image(x):
    # Get the min and max values for all pixels in the input.
    x_min = x.min()
    x_max = x.max()

    # Normalize so all values are between 0.0 and 1.0
    x_norm = (x - x_min) / (x_max - x_min)

    return x_norm


def plot_image(image):
    # Normalize the image so pixels are between 0.0 and 1.0
    img_norm = normalize_image(image)

    # Plot the image.
    plt.imshow(img_norm, interpolation='nearest')
    plt.show()


def plot_images(images, show_size=100):
    # The show_size is the number of pixels to show for each image.
    # The max value is 299.

    # Create figure with sub-plots.
    fig, axes = plt.subplots(2, 3)

    # Adjust vertical spacing.
    fig.subplots_adjust(hspace=0.1, wspace=0.1)

    # Use interpolation to smooth pixels?
    smooth = True

    # Interpolation type.
    if smooth:
        interpolation = 'spline16'
    else:
        interpolation = 'nearest'

    # For each entry in the grid.
    for i, ax in enumerate(axes.flat):
        # Get the i'th image and only use the desired pixels.
        img = images[i, 0:show_size, 0:show_size, :]

        # Normalize the image so its pixels are between 0.0 and 1.0
        img_norm = normalize_image(img)

        # Plot the image.
        ax.imshow(img_norm, interpolation=interpolation)

        # Remove ticks.
        ax.set_xticks([])
        ax.set_yticks([])

    # Ensure the plot is shown correctly with multiple plots
    # in a single Notebook cell.
    plt.show()


# Helper-function for optimizing and plotting images
def optimize_images(conv_id=None, num_iterations=30, show_size=100):
    # Figure 6 images that maximize the 6 first features in the layer
    # given by the conv_id.
    # Parameters:
    # conv_id:  Integer identifying the convolutional layer to
    #           maximize. It is an index into conv_names.
    #           If None then use the last layer before the softmax output.
    # num_iterations: Number of optimization iterations to perform.
    # show_size: Number of pixels to show for each iamge. Max 299.

    # Which layer are we using?
    if conv_id is None:
        print("Final fully-connected layer before softmax.")
    else:
        print("Layer: {}".format(conv_names[conv_id]))

    # Initialize the array of images.
    images = []

    # For each feature do the following. Note that the
    # last fully-connected layer only supports numbers
    # between 1 and 1000, while the convolutional layers
    # support numbers between 0 and some other number.
    # So we just use the numbers between 1 and 7.
    for feature in range(1, 7):
        print("Optimizing image for feature no. {}".format(feature))

        # Find the image that maximizes the given feature
        # for the network layer identified by conv_id (or None).
        image = optimize_image(conv_id=conv_id, feature=feature,
                               show_progress=False,
                               num_iterations=num_iterations)

        # Squeeze the dim of the array.
        image = image.squeeze()

        # Append to the list of images.
        images.append(image)

    # Convert to numpy-array so we can index all dimensions easily.
    images = np.array(images)

    # Plot the images.
    plot_images(images=images, show_size=show_size)

# Results
# Optimize a single image for an early convolutional layer
print('\nconv_id[5]: {}\n'.format(conv_names[5]))
image = optimize_image(conv_id=5,
                       feature=2,
                       num_iterations=30,
                       show_progress=True)

plot_image(image)

# Optimize multiple images for convolutional layers
optimize_images(conv_id=0, num_iterations=10)

optimize_images(conv_id=3, num_iterations=30)

optimize_images(conv_id=4, num_iterations=30)

optimize_images(conv_id=5, num_iterations=30)

optimize_images(conv_id=6, num_iterations=30)

optimize_images(conv_id=7, num_iterations=30)

optimize_images(conv_id=8, num_iterations=30)

optimize_images(conv_id=9, num_iterations=30)

optimize_images(conv_id=10, num_iterations=30)

optimize_images(conv_id=20, num_iterations=30)

optimize_images(conv_id=30, num_iterations=30)

optimize_images(conv_id=40, num_iterations=30)

optimize_images(conv_id=50, num_iterations=30)

optimize_images(conv_id=60, num_iterations=30)

optimize_images(conv_id=70, num_iterations=30)

optimize_images(conv_id=80, num_iterations=30)

optimize_images(conv_id=90, num_iterations=30)

optimize_images(conv_id=93, num_iterations=30)

# Final fully-connected layer before Softmax
optimize_images(conv_id=None, num_iterations=30)

image = optimize_image(conv_id=None, feature=1, num_iterations=100, show_progress=True)
plot_image(image=image)

# Close TensorFlow Session
