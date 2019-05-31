# Inports
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

# Functions and classes for loading and using the Inception model.
import inception

print('TensorFlow version: {}'.format(tf.__version__))

# Inception Model
# Download the Inception model from the internet
inception.data_dir = '../inception/'
inception.maybe_download()

# Load the Inception Model
model = inception.Inception()

# Get Input and Output for the Inception Model
resized_image = model.resized_image
y_pred = model.y_pred
y_logits = model.y_logits


# Hack the Inception Model
# Set the graph for the Inception model as the default graph,
# so that all changes inside this with-block are done to that graph.
with model.graph.as_default():
    # Add a placeholder variable for the target class-number.
    # This will be set to e.g. 300 for the 'bookcase' class.
    pl_cls_target = tf.placeholder(dtype=tf.int32)

    # Add a new loss-function. This is the cross-entropy.
    # See Tutorial #01 for an explanation of corss-entropy.
    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y_logits, labels=[pl_cls_target])

    # Get the gradient for the loss-function with regard to
    # the resized input image.
    gradient = tf.gradients(ys=loss, xs=resized_image)

# TensorFlow Session
session = tf.Session(graph=model.graph)


# Helper-function for finding Adversary Noise
def find_adversary_noise(image_path, cls_target, noise_limit=3.0, required_score=0.99, max_iterations=100):
    # Find the noise that must be added to the given image so
    # that it is classified as the target-class.

    # image_path: File-path to the input-image (must bt *.jpg).
    # cls_target: Target class-number (integer between 1-1000).
    # noise_limit: Limit for pixel-values in the noise.
    # required_score: Stop when target-class score reaches this.
    # max_iterations: Max number of optimization iterations to perform.

    # Create a feed-dict with the image.
    feed_dict = model._create_feed_dict(image_path=image_path)

    # Use TensorFlow to calculate the predicted class-cores (aka. probabilities) as well as the resized iamge.
    pred, image = session.run([y_pred, resized_image], feed_dict=feed_dict)

    # Convert to one-dimensional array.
    pred = np.squeeze(pred)

    # Predicted clas-number.
    cls_source = np.argmax(pred)

    # Score for the predicted class (aka. probability or confidence).
    score_source_org = pred.max()

    # Names for the source and target classes.
    name_source = model.name_lookup.cls_to_name(cls_source, only_first_name=True)
    name_target = model.name_lookup.cls_to_name(cls_target, only_first_name=True)

    # Initialize the noise to zero.
    noise = 0
    noisy_image, score_source, score_target = None, None, None

    # Perform a number of optimization iterations to find
    # the noise that causes mis-clasification of the input image.
    for i in range(max_iterations):
        print("Iteration:", i)

        # The noise image is just the sum of the input iamge and noise.
        noisy_image = image + noise

        # Create a feed-dict. This feeds the noisy image to the
        # tensor in the graph that holds the resized image, because
        # this is the final stage for inputting raw image data.
        # This als feeds the target class-number that we desire.
        feed_dict = {model.tensor_name_resized_image: noisy_image,
                     pl_cls_target: cls_target}

        # Calculate the predeicted class-scores as well as the gradient.
        pred, grad = session.run([y_pred, gradient], feed_dict=feed_dict)

        # Convert the predicted class-scores to a one-dim array.
        pred = np.squeeze(pred)

        # The scores (probabilities) for the source and target classes.
        score_source = pred[cls_source]
        score_target = pred[cls_target]

        # Squeeze the dimensionality for the gradient-array.
        grad = np.array(grad).squeeze()

        # The gradient now tells us how much we need to change the
        # noisy input image in order to move the predicted class
        # closer to the desired target-class.

        # Calculate the max of the absolute gradient values.
        # This is used to calculate the step-size.
        grad_absmax = np.abs(grad).max()

        # If the gradient is very small then use a lower limit,
        # because wi will use it as a divisor.
        if grad_absmax < 1e-10:
            grad_absmax = 1e-10

        # Calculate the step-size for updating the iamge-noise.
        # This ensures that at least one pixel colour is changed by 7.
        # Recal lthat pixel colours can have 255 different values.
        # This step-size was found to give fast convergence.
        step_size = 7. / grad_absmax

        # Pritn the score etc. for the source-class.
        msg = "Source score: {0:>7.2%}, class-number: {1:>4}, class-name: {2}"
        print(msg.format(score_source, cls_source, name_source))

        # Print the score etc. for the target-class.
        msg = "Target score: {0:7.2%}, class-number: {1:>4}, class-name: {2}"
        print(msg.format(score_target, cls_target, name_target))

        # Print statistics for the gradietn.
        msg = "Gradient min: {0:>9.6f}, max: {1:>9.6f}, stepsize: {2:>9.2f}"
        print(msg.format(grad.min(), grad.max(), step_size))

        # Newline.
        print()

        # If the scor for the target-class is not high enouth.
        if score_target < required_score:
            # Update the image-noise by subtracting the gradient
            # scaled by the step-size.
            noise -= step_size * grad

            # Ensure the noise is whithin the desired range.
            # This avoids distorting the image too much.
            noise = np.clip(a=noise, a_min=-noise_limit, a_max=noise_limit)

        else:
            # Abort the optimizatio nbecause the score is high enough.
            break

    return image.squeeze(), noisy_image.squeeze(), noise, \
           name_source, name_target, score_source, score_source_org, score_target


# Helper-function for plotting image and noise
def normalize_image(x):
    # Get the min and max values for all pixels in the input.
    x_min = x.min()
    x_max = x.max()

    # Normalize so all values are between 0.0 and 1.0
    x_norm = (x - x_min) / (x_max - x_min)

    return x_norm


def plot_images(image, noise, noisy_image, name_source, name_target, score_source, score_source_org, score_target):
    # Plot hte image, the noisy image and the noise.
    # Also shows the class-names and scores.
    #
    # Note that the noise is amplified th use the full range of colours, otherwise if the noise is very low it
    # would be hard to see.
    # image: Original input image.
    # noise: Noise that has been added to the image.
    # noisy_image: Input image + noise.
    # name_source: Name of the source-class.
    # name_target: Name of the target-class.
    # score_source: Score for the source-class.
    # score_source_org: Original score for the source-class.
    # score_target: Score for the target-clas.

    # Create figure with sub-plots.
    fig, axes = plt.subplots(1, 3, figsize=(10, 10))

    # Adjust vertical spacing
    fig.subplots_adjust(hspace=0.1, wspace=0.1)

    # Use interpolation to smooth pixels?
    smooth = True

    # Interpolation type.
    if smooth:
        interpolation = 'spline16'
    else:
        interpolation = 'nearest'

    # Plot the original iamge.
    # Note that the pixel-values are normalized to the [0.0, 1.0]
    # rang eby dividing with 255.
    ax = axes.flat[0]
    ax.imshow(image / 255., interpolation=interpolation)
    msg = "Original Image:\n{0} ({1:.2%})"
    xlabel = msg.format(name_source, score_source_org)
    ax.set_xlabel(xlabel)

    # Plot the noisy image.
    ax = axes.flat[1]
    ax.imshow(noisy_image / 255., interpolation=interpolation)
    msg = "Image + Noise:\n{0} ({1:.2%})\n{2} ({3:.2%})"
    xlabel = msg.format(name_source, score_source, name_target, score_target)
    ax.set_xlabel(xlabel)

    # Plot the noise.
    # The colours are amplified otherwise they would be hard to see.
    ax = axes.flat[2]
    ax.imshow(normalize_image(noise), interpolation=interpolation)
    xlabel = "Amplified Noise"
    ax.set_xlabel(xlabel)

    # Remove ticks from all the plots.
    for ax in axes.flat:
        ax.set_xticks([])
        ax.set_yticks([])

    # Ensure the plot is shown correctly with multiple plots
    # in a single Notebook cell.
    plt.show()


# Helper-function for finding and plotting adversarial example
def adversary_example(image_path, cls_target, noise_limit, required_score):
    # Find and plot adversarial noise for the given image.
    #
    # image_path: File-path to the input-image (must be *.jpg).
    # cls_target: Target class-number (integer between 1-1000).
    # noise_limit: Limit for pixel-values in the noise.
    # required_score: Stop when target-class score reaches this.

    # Find the adversarial noise.
    image, noisy_image, noise, name_source, name_target, score_source, score_source_org, score_target = \
        find_adversary_noise(image_path=image_path,
                             cls_target=cls_target,
                             noise_limit=noise_limit,
                             required_score=required_score)

    # Plot the image and the noise.
    plot_images(image=image, noise=noise, noisy_image=noisy_image,
                name_source=name_source, name_target=name_target,
                score_source=score_source, score_source_org=score_source_org, score_target=score_target)

    # Print some statistics for the noise.
    msg = "Noise min: {0:.3f}, max: {1:.3f}, mean: {2:.3f}, std: {3:.3f}"
    print(msg.format(noise.min(), noise.max(), noise.mean(), noise.std()))

# Results
# Parrot
image_path = "../images/parrot_cropped1.jpg"
adversary_example(image_path=image_path,
                  cls_target=300,
                  noise_limit=3.0,
                  required_score=0.99)

# Elon Musk
image_path = "../images/elon_musk.jpg"
adversary_example(image_path=image_path,
                  cls_target=300,
                  noise_limit=3.0,
                  required_score=0.99)

# Willy Wonka (New)
image_path = "../images/willy_wonka_new.jpg"
adversary_example(image_path=image_path,
                  cls_target=300,
                  noise_limit=3.0,
                  required_score=0.99)

# Willy Wonka (Old)
image_path = "../images/willy_wonka_old.jpg"
adversary_example(image_path=image_path,
                  cls_target=300,
                  noise_limit=3.0,
                  required_score=0.99)

# Close TensorFlow Session
# This has been commented out in case you want to modify and experiment
# with the Notebook without having to restart it.
session.close()
model.close()
