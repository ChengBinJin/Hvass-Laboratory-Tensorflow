# Imports
import os
import PIL
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from tensorflow.python.keras.models import Model, Sequential
from tensorflow.python.keras.layers import Dense, Flatten, Dropout
from tensorflow.python.keras.applications import VGG16
from tensorflow.python.keras.applications.vgg16 import preprocess_input, decode_predictions
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.keras.optimizers import Adam, RMSprop

print('- TensorFlow version: {}'.format(tf.__version__))

# Helper Functions
# Helper-function for joining a directoy and list of filenames
def path_join(dirname, filenames):
    return [os.path.join(dirname, filename) for filename in filenames]


# Helper-function for loading imagess
def load_images(image_paths):
    # Load the images from disk.
    images = [plt.imread(path) for path in image_paths]

    # Convert to a numpy array and return it.
    return np.asarray(images)


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
        # There may be less than 9 images, ensure it doesn't crash.
        if i < len(images):
            # Plot image.
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


# Dataset: Knifey-Spoony
import knifey

knifey.maybe_download_and_extract()
knifey.copy_files()

train_dir = knifey.train_dir
test_dir = knifey.test_dir

# Pre-trained Model: VGG16
model = VGG16(include_top=True, weights='imagenet')

# Input Pipeline
input_shape = model.layers[0].output_shape[1:3]
print('Input shape: {}'.format(input_shape))

datagen_train = ImageDataGenerator(
    rescale=1./255,
    rotation_range=180,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=[0.9, 1.5],
    horizontal_flip=True,
    vertical_flip=True,
    fill_mode='nearest'
)

datagen_test = ImageDataGenerator(rescale=1./255)

batch_size = 20

is_save_img = True
if is_save_img:
    save_to_dir = None
else:
    save_to_dir = 'augmented_images/'

generator_train = datagen_train.flow_from_directory(directory=train_dir,
                                                    target_size=input_shape,
                                                    batch_size=batch_size,
                                                    shuffle=True,
                                                    save_to_dir=save_to_dir)

generator_test = datagen_test.flow_from_directory(directory=test_dir,
                                                  target_size=input_shape,
                                                  batch_size=batch_size,
                                                  shuffle=False)

steps_test = generator_test.n / batch_size
print('steps_test: {}'.format(steps_test))

image_paths_train = path_join(train_dir, generator_train.filenames)
image_paths_test = path_join(test_dir, generator_test.filenames)

cls_train = generator_train.classes
cls_test = generator_test.classes

class_names = list(generator_train.class_indices.keys())
print('class_names: {}'.format(class_names))

num_classes = generator_train.num_classes
print('num_classes: {}'.format(num_classes))


# Plot a few images to see if data is correct
# Load the first images from the train-set.
images = load_images(image_paths=image_paths_train[0:9])

# Get the true classes for those images.
cls_true = cls_train[0:9]

# Plot the images and labels using helper-function above.
plot_images(images=images, cls_true=cls_true, smooth=True)


# Class Weights
from sklearn.utils.class_weight import compute_class_weight

class_weight = compute_class_weight(class_weight='balanced',
                                    classes=np.unique(cls_train),
                                    y=cls_train)
print('class_weight: {}'.format(class_weight))
print('class_names: {}'.format(class_names))


# Helper-function for printing confusion matrix
# Import a function from sklearn to calculate the confusion-matrix.
from sklearn.metrics import confusion_matrix

def print_confusion_matrix(cls_pred):
    # cls_pred is an array of the predicted class-number for
    # all images in the test-set.

    # Get the confusion matrix using sklearn.
    cm = confusion_matrix(y_true=cls_test,  # True class for test-set.
                          y_pred=cls_pred)  # Predicted class.

    print("Confusion matrix:")

    # Print the confusion matrix as text.
    print(cm)

    # Print the class-names for easy reference.
    for i, class_name in enumerate(class_names):
        print("({0}) {1}".format(i, class_name))


# Helper-function for plotting example errors
def plot_example_errors(cls_pred):
    # cls_pred is an array fo the predicted class-number for
    # all images in the test-set.

    # Boolean array whether the predicted class is incorrect.
    incorrect = (cls_pred != cls_test)

    # Get the file-paths for images that were incorrectly classified.
    image_paths = np.array(image_paths_test)[incorrect]

    # Load the first 9 images
    images = load_images(image_paths=image_paths[0:9])

    # Get the predicted classes for those iamges.
    cls_pred = cls_pred[incorrect]

    # Get the true classes for those images.
    cls_true = cls_test[incorrect]

    # Plot the 9 images we have loaded and their corresponding classes.
    # We have only loaded 9 images so there is no need to slice those again.
    plot_images(images=images,
                cls_true=cls_true[0:9],
                cls_pred=cls_pred[0:9])


def example_errors():
    # The Keras data-generator for the test-set must be reset
    # before processing. This is because the generator will loop
    # infinitely and keep an internal index into the dataset.
    # So it might start in the middle of the test-set if we do
    # not reset if first. This makes it impossible to match the
    # predicted classes with the input images.
    # If we reset the generator, then it always starts at the
    # beginnig so we know exactly which input-images were used.
    generator_test.reset()

    # Predict the classes for all images in the test-set.
    y_pred = new_model.predict_generator(generator_test, steps=steps_test)

    # Convert the predicted classes from arrays to integers.
    cls_pred = np.argmax(y_pred, axis=1)

    # Plot examlpes of mis-classified images.
    plot_example_errors(cls_pred)

    # Print the confusion matrix.
    print_confusion_matrix(cls_pred)


# Helper-function for plotting training history
def plot_training_history(history):
    # Get the classification accuracy and loss-value
    # for the training-set.
    acc = history.history['categorical_accuracy']
    loss = history.history['loss']

    # Get if for the validation-set (we only use the test-set).
    val_acc = history.history['val_categorical_accuracy']
    val_loss = history.history['val_loss']

    # Plot the accuracy and loss-values for the training-set.
    plt.plot(acc, linestyle='-', color='b', label='Training Acc.')
    plt.plot(loss, 'o', color='b', label='Training Loss')

    # Plot it for the test-set.
    plt.plot(val_acc, linestyle='--', color='r', label='Test Acc.')
    plt.plot(val_loss, 'o', color='r', label='Test Loss')

    # Plot title and legend.
    plt.title('Training and Test Accuracy')
    plt.legend()

    # Ensure the plot shows correctly.
    plt.show()


# Example Predictions
def predict(image_path):
    # Load and resize the image using PIL.
    img = PIL.Image.open(image_path)
    img_resized = img.resize(input_shape, PIL.Image.LANCZOS)

    # Plot the image.
    plt.imshow(img_resized)
    plt.show()

    # Convert the PIL iamge to a nupy-array with the proper shape.
    img_array = np.expand_dims(np.array(img_resized), axis=0)

    # Use the VGG16 model to make a prediction.
    # This outputs an array with 1000 numbers corresponding to
    # the classes of the ImageNet-dataset.
    pred = model.predict(img_array)

    # Decode the output of the VGG16 model.
    pred_decoded = decode_predictions(pred)[0]

    # Print the predictions.
    for code, name, score in pred_decoded:
        print("{0:>6.2%} : {1}".format(score, name))

predict(image_path='../images/parrot_cropped1.jpg')
predict(image_path=image_paths_train[0])
predict(image_path=image_paths_train[1])
predict(image_path=image_paths_test[0])

# Transfer Learning
model.summary()
transfer_layer = model.get_layer('block5_pool')
print('transfer_layer output: {}'.format(transfer_layer.output.shape))

conv_model = Model(inputs=model.input,
                   outputs=transfer_layer.output)
# Start a new Keras sequential model.
new_model = Sequential()

# Add the convolutional part of the VGG16 model from above.
new_model.add(conv_model)

# Flatten the output of the VGG16 model because it is from
# a convolutional layer
new_model.add(Flatten())
new_model.add(Flatten())

# Add a dense (aka. fully-conncected) layer.
# THis is for combining features that the VGG16 model has
# recognized in the image.
new_model.add(Dense(1024, activation='relu'))

# Add a dropout-layer which may prevent overfitting and
# imporve generalization ability to unseen data e.g. the test-set.
new_model.add(Dropout(0.5))

# Add the final layer for the actual classification.
new_model.add(Dense(num_classes, activation='softmax'))

optimizer = Adam(lr=1e-5)

loss = 'categorical_crossentropy'
metrics = ['categorical_accuracy']

def print_layer_trainable():
    for layer in conv_model.layers:
        print("{0}:\t{1}".format(layer.trainable, layer.name))

print_layer_trainable()

conv_model.trainable = False

for layer in conv_model.layers:
    layer.trainable = False

print_layer_trainable()

new_model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

epochs = 20
steps_per_epoch = 100

history = new_model.fit_generator(generator=generator_train,
                                  epochs=epochs,
                                  steps_per_epoch=steps_per_epoch,
                                  class_weight=class_weight,
                                  validation_data=generator_test,
                                  validation_steps=steps_test)

plot_training_history(history)

result = new_model.evaluate_generator(generator_test, steps=steps_test)
print("Test-set clasification accuracy: {0:.2%}".format(result[1]))

example_errors()

# Fine-Tuning
conv_model.trainable = True

for layer in conv_model.layers:
    # Boolean whtether this layer is trainable.
    trainable = ('block5' in layer.name or 'block4' in layer.name)

    # Set the Lay's boll
    layer.trainable = trainable

optimizer_fine = Adam(lr=1e-7)
new_model.compile(optimizer=optimizer_fine, loss=loss, metrics=metrics)

history = new_model.fit_generator(generator=generator_train,
                                  epochs=epochs,
                                  steps_per_epoch=steps_per_epoch,
                                  class_weight=class_weight,
                                  validation_data=generator_test,
                                  validation_steps=steps_test)

plot_training_history(history)

result = new_model.evaluate_generator(generator_test, steps=steps_test)
print("Test-set classification accuracy: {0:.2%}".format(result[1]))

example_errors()


