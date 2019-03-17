# Imports
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

# from IPython.display import Image, display

# Functions and classes for loading and using the Inception model.
import inception

print("TensorFlow version: {}".format(tf.__version__))

# Download the Inception Model
inception.maybe_download()

# Load the Inceptin Model
model = inception.Inception()

# Helper-function for classifying and plotting images
def classify(image_path):
    # Display the image.
    img = cv2.imread(image_path)
    # print('img shape: {}'.format(img.shape))

    print('\nimage path: {}'.format(image_path))
    cv2.imshow('Image', img)

    # Use the Inception model to classify the image.
    pred = model.classify(image_path=image_path)

    # Print the scores and names for the top-10 predictions.
    model.print_scores(pred=pred, k=10, only_first_name=True)

    cv2.waitKey(0)

# Panda
image_path = os.path.join(inception.data_dir, 'cropped_panda.jpg')
classify(image_path)

# Parrot (Original Image)
classify(image_path="../images/parrot.jpg")

# Parrot (Resized Image)
def plot_resized_image(image_path):
    # Get the resized image from tne Inception model.
    resized_image = model.get_resized_image(image_path=image_path)

    # Plot the image.
    plt.imshow(resized_image, interpolation='nearest')

    # Ensure that the plot is shown.
    plt.show()

plot_resized_image(image_path="../images/parrot.jpg")

# Parrot (Cropped Image, Top)
classify(image_path='../images/parrot_cropped1.jpg')

# Parrot (Cropped Image, Middle)
classify(image_path='../images/parrot_cropped2.jpg')

# Parrot (Cropped Image, Bottom)
classify(image_path='../images/parrot_cropped3.jpg')

# Parrot (Paded Image)
classify(image_path='../images/parrot_padded.jpg')

# Elon Musk (299 x 299 pixels)
classify(image_path='../images/elon_musk.jpg')

# Elon Musk (100 x 100 pixels)
classify(image_path='../images/elon_musk_100x100.jpg')

# Willy Wonka (Gene Wilder)
classify(image_path='../images/willy_wonka_old.jpg')

# Willy wonka (Johnny Depp)
classify(image_path='../images/willy_wonka_new.jpg')

# Close TensorFlow Session
model.close()
