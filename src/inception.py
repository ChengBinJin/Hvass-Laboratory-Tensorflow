##########################################################################################
#
# The Inception Model v3 for TensorFlow.
#
# This is a pre-trained Deep Neural Network for classifying images.
# You provide an image or filename for a jpeg-file which will be loaded and input to
# the Inception model, which will then output an array of numbers indicating how
# likely it is taht the input-image is of each class.
#
# See the example code at the bottom of this file or in the accompanying Python Notebooks.
#
# Tutorial #07 shows how to use the Inception model.
# Tutorial #08 shows how to use it for Transfer Learning.
#
# What is Transfer Learning?
#
# Transfer Learning is the use of a Neural Network for classifying images from another
# data-set than it was trained on. For example, the Inception model was trained on the
# ImageNet data-set using a very powerful and expensive computer. But the Inception model
# can be re-used on data-sets it was not trained on without having to re-train the entire
# model, even though the number of classes are different for the two data-sets. This
# allows you to use the Inception model on your own data-sets without the need for a
# very powerful and expensive computer to train it.
#
# The last layer of the Inception model before the softmax-classifier is called the
# Transfer Layer because the output of that layer will be used as the input in your new
# softmax-classifier (or as the input for another neural network), which will then be
# trained on your own data-set.
#
# The output values of the Transfer Layer are called Transfer Values. These are the
# actual values that will be input to your new softmax-classifier or to another neural
# network that you create.
#
# The word 'bottleneck' is also sometimes used to refer to the Transfer Layer or Transfer
# Values, but it is a confusing word that is not used here.
#
# Implemented in Python 3.6 with TensorFlow v1.12.0rc0
##########################################################################################
