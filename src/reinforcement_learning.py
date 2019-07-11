##################################################################################
#
# Reinforcement Learning (Q-Learning) for Atari Games
#
# How to run:
#
# To train a Neural Network for playing the Atari game Breakout,
# run the following command in a terminal window.
#
# python reinforcement_learning.py --env 'Breakout-v0' --training
#
# The agent should start to improve after a few hours, but a full
# training run required 150 hours on a 2.6 GHz CPU and GTX 1070 GPU.
#
# The hyper-parameters were tuned for Breakout and did not work
# quite as well for SpaceInvaders. Can you find better parameters?
#
# Once the Neural Network has been trained, you can test it and
# watch it play the game by running this command in the terminal:
#
# python reinforcement_learning.py --env 'Breakout-v0' --render --episodes 2
#
# Requirements:
#
# - Python 3.6 (Python 2.7 may not work)
# - TensorFlow 1.14.0
# - OpenAI Gym 0.13.1
#
# Summary:
#
# This program implements a variatn of Reinforcement Learning known as
# Q-learning. Imagine that we have an agent hat must take actions in
# some environment so as to maximize the cumulative reward over its life.
# The agent sees the state of the game-environment through images
# which are sent through a Neural network in TensorFlow, so as to
# estimate which action is most likely to maximize the cumulative
# reward of all future actions. These action-valeus are also called
# Q-values. If the Q-values ar eknows in advance, then the agent merely
# has to select the action corresponding to the highest Q-value in
# each state of the game. But the Q-values are not known in advance
# and must be learnt while the agent is playing the game.
# This is done by initializing all Q-values to zero and then having
# the agent take random actions. Whenever the agent obtains a reward,
# the estimated Q-values can be updated with the new information.
# The agent gradually learns to play the game better and better
# because the Neural network becomes better at estimating the Q-values.
# But this process is very slow and the basic algorithm implemented
# here typically requires 100 million steps in the game-environment,
# although it will typically start to show improvement much sooner.
#
# Main classes:
#
# - MoitionTracer:
#
#   This takes raw images from the game-environment and processes them.
#   The output is called a state and consists of two images of equal size:
#   (1) The last image from the game-environment, resized and gray-scaled.
#   (2) A motion-trace that shows the recent trajectories of objects.
#
# - ReplayMemory:
#
#   Successive image-frames of the game-environment are almost identical.
#   If we train the Neural network to estimate Q-values from a small
#   number of successive image-frames, then it cannot learn to distinguish
#   important features and the trainng becomes unstable. For the basic
#   Q-learning algorithm we need many thousand states from the game-environment
#   in order to learn import features so the Q-values can be estimated.
#
# - NeuralNetwork:
#
#   This implements a Neural Network for estimating Q-values. It takes as
#   input a state of the game-environment that was output by the Moition Tracer,
#   and then the Neural Network outputs the estimated Q-values that indicate
#   the cumulative reward of taking each action for a given state of the game.
#
# - Agent:
#
#   This implements the agent that plays games. it loads an Atari-game from
#   OpenAI Gym and inputs the game-images to the Motion Tracer, which in turn
#   outputs a state that is input to the Neural Network, which estimates the
#   Q-values that are used for selecting the next action. The agent then
#   takes a step in the game-environment. During training, the data is added
#   to the Replay Memory and when it is sufficiently full, an optimization run
#   is performed so as to improvethe Neural network's ability to estimate
#   Q-values. This procedure is repeated many, many times until the Neural
#   Network is sufficiently accurate at esttimating Q-values.
#
# The Q-Value Formula:
#
# The Q-values for a given state is a vector with a value for each possible
# action, indicating the total future reward that can be had by taking each
# action. The Q-values are initialized to roughly zero and must then be
# imporved iteratively when new information becomes availabel.
#
# We know which action was taken in the current step and what the observed
# reward was, so the estimated Q-value can be improved with this information.
# The Q-value estimates the total cumulative reward for all future steps, which
# is why we use the max Q-value for the next step.
#
# The formula for updating Q-values is implemented in the ReplayMemory-class
# in the function update_all_q_values(), which does a complete backwards-sweep
# through the Replay Memory. The formula for updating the Q-values is:
#
# Q-value for this state and action = observed reward for the current step
#                               + discount factor * max Q-value for next step
#
# The discount factor is a number slightly below 1.0 (e.g. 0.97) which causes
# distant future rewards to have a smaller effect on the Q-values. This means
# that if the reward is the same, then it is considered more valuable to get
# the reward sooner rather than later.
#
# Pseudo-Code:
#
# There are many lines of source-code required to implement all this, but the
# main ideas of the algorithm can be described more simply in pseudo-code:
#
# 1) Initialize all Q-values to roughly zero.
#    We use a Neural network to estimate the Q-values, so this means
#    we have to initialize the Neural Network with small random weights.
#
# 2) Reset the game-environment and Motion Tracer.
#
# 3) Get the state from the Motion Tracer which consists of two gray-scale
#    images. The first is the image of the game-environment and the second
#    is a motion-trace showing recent movements in the game-environment.
#
# 4) Input the state to the Neural Network to estiamte the Q-values.
#
# 5) Either take a random action with probability epsilon, or take the
#    action with the highest Q-value. This is called the epsilon-greedy policy.
#
# 6) Add the state, action and observed reward to the Replay Memory.
#
# 7) When the Replay Memory is sufficiently fully, first perform a full
#    backwards-sweep to update all the Q-values with the ovserved reward.
#
#    Then perform an optimization run of the Neural Network.
#    This takes random batches of data from the Replay Memory and uses them
#    for training the Neural Network to become better at estimating Q-values.
#
#    Save a checkpoint for the Neural network so we can reload it later.
#
# 8) Input the recent image of the game-environment to the Motion Tracer
#    and repeat from step (3).
#
##################################################################################
#
# This file is part of the TensorFlow Tutorials available at:
#
# https://github.com/Hvass-Labs/TensorFlow-Tutorials
#
# Published under the MIT License. See the file LICENSE for detailes.
#
# Copyright 2017 by Magnus Erik Hvass Pedersen
#
##################################################################################

import os
import sys
import time
import csv
import scipy
import argparse
import gym
import numpy as np
import tensorflow as tf

##################################################################################
# File-paths are global variables for convenience so they don't
# have to be passed around between all the objects.
# You should first set checkpoint_base_dir to whichever you like,
# then call the function update_paths(env_name) to update all the paths.
# This should be done before you create the Agent and NeuralNetwork etc.

# Default base-directory for the checkpoints and log-files,
# The environment-name will be appended to this.
checkpoint_base_dir = "../checkpoints_tutorial16"

# Combination of base-dir and environment-name.
checkpoint_dir = None

# Full path for the log-file for rewards.
log_reard_path = None

# Full path for the log-file for Q-values.
log_q_values_path = None


def update_paths(env_name):
    """
    Update the path-names for the checkpoint-dir and log-files.

    Call this after you have changed checkpoint_base_dir and
    before you create the Neural Network.

    :param env_name:
        Name of the game-environment you will use in OpenAI Gym.
    """

    global checkpoint_dir
    global log_reward_path
    global log_q_values_path

    # Add teh environment-name to the checkpoint-dir.
    checkpoint_dir = os.path.join(checkpoint_base_dir, env_name)

    # Create the checkpoint-dir if it does not already exist.
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    # File-path for the log-file for episode rewards.
    log_reward_path = os.path.join(checkpoint_dir, "log_reward.txt")

    # File-path for the log-file for Q-values.
    log_q_values_path = os.path.join(checkpoint_dir, "log_q_values.txt")

##################################################################################
# Classes used for logging data during training.


class Log:
    """
    Base-class for logging data to a text-file during training.

    It is possible to use TensorFlow / TensorBoard for this,
    but it is quite awkward to implement, as it was intended
    for logging variables and otehr aspects of the TensorFlow graph.
    We want to log the reward and Q-values which are not in that graph.
    """

    def __init__(self, file_path):
        # Set the path for the log-file.
        # Nothing is saved or loaded yet.

        # Path forthe log-file.
        self.file_path = file_path

        # Data to be read from the log-file by the _read() function.
        self.count_episodes = None
        self.count_states = None
        self.data = None

    def _write(self, count_episodes, count_states, msg):
        """
        Write a line to the log-file. This is only called by sub-classes.

        :param count_episodes:
            Counter for the number of episodes processed during training.

        :param count_states:
            Counter for the number of states processed during training.

        :param msg:
            Message to write in the log.
        """

        with open(file=self.file_path, mode='a', buffering=1) as file:
            msg_annotated = "{0}\t{1}\t{2}\n".format(count_episodes, count_states, msg)
            file.write(msg_annotated)

    def _read(self):
        """
        Read the log-file into memory so it can be plotted.

        It sets self.count_episodes, self.count_states and self.data
        """

        # Open and read the log-file.
        with open(self.file_path) as f:
            reader = csv.reader(f, delimiter="\t")
            self.count_episodes, self.count_states, *data = zip(*reader)

        # Convert the remaining log-data to a NumPy float-array.
        self.data = np.array(data, dtype='float')


class LogReward(Log):
    # Log the rewards obtained for episodes during training.

    def __init__(self):
        # These will be set in read() below.
        self.episode = None
        self.mean = None

        # Super-class init.
        Log.__init__(self, file_path=log_reward_path)

    def write(self, count_episodes, count_states, reward_episode, reward_mean):
        """
        Write the episode and mean reward to file.

        :param count_episodes:
            Counter for the number of episodes processed during training.

        :param count_states:
            Counter for the number of states processed during training.

        :param reward_episode:
            Reward for one episode.

        :param reward_mean:
            Mean reward for the last e.g. 30 episodes.
        """

        msg = "{0:.1f}\t{1:.1f}".format(reward_episode, reward_mean)
        self._write(count_episodes=count_episodes, count_states=count_states, msg=msg)

    def read(self):
        """
        Read the log-file into memory so it can be plotted.

        It sets self.count_episodes, self.count_states, self.episode and self.mean
        """

        # Read the log-file using the upser-class.
        self._read()

        # Get the episode reward.
        self.episode = self.data[0]

        # Get the mean reward.
        self.mean = self.data[1]


class LogQValues(Log):
    # Log the Q-Values during training.

    def __init__(self):
        # These will be set in read() below.
        self.min = None
        self.mean = None
        self.max = None
        self.std = None

        # Super-class init.
        Log.__init__(self, file_path=log_q_values_path)

    def write(self, count_episodes, count_states, q_values):
        """
        Write basic statistics for the Q-values to file.

        :param count_episodes:
            Counter for the number of episodes processed during training.

        :param count_states:
            Counter for the number of states processed during training.

        :param q_values:
            Numpy array with Q-values from the replay-memory.
        """

        msg = "{0:.3f}\t{1:.3f}\t{2:.3f}\t{3:.3f}".format(np.min(q_values),
                                                          np.mean(q_values),
                                                          np.max(q_values),
                                                          np.std(q_values))

        self._write(count_episodes=count_episodes,
                    count_states=count_states,
                    msg=msg)

    def read(self):
        """
        Read the log-file into memory so it can be plotted.

        It sets self.count_episodes, self.count_states, self.min / mean / max / std.
        """

        # Read the log-file using the super-class.
        self._read()

        # Get the logged statistics for the Q-values.
        self.min = self.data[0]
        self.mean = self.data[1]
        self.max = self.data[2]
        self.std = self.data[3]

#######################################################################################


def print_progress(msg):
    """
    Print progress on a single line and overwrite the line.
    Used during optimization.
    """

    sys.stdout.write("\r" + msg)
    sys.stdout.flush()

#######################################################################################
# A state is basically just a multi-dimensional array that is being
# input to the Neural Network. The state consists of pre-processed images
# from the game-environment. We will just convert the game-images to
# gray-scale and resize them to roughly half their size. This is mainly
# so we can save memory-space in the Replay memory further below.
# The oroginal DeepMind paper used game-states consisting of 4 frames of
# game-images that were gray-scaled, resized to 110 x 84 pixels, and then
# cropped to 84 x 84 pixels because their implementation only supported this.

# Height of each image-frame in the state.
state_height= 105

# Width of each image-frame in the state.
state_width = 80

# Size of each image in the state.
state_img_size = np.array([state_height, state_width])

# Number of images in the state.
state_channels = 2

# Shape of the state-array
state_shape = [state_height, state_width, state_channels]

#######################################################################################
# Functions and classes for processing images from the game-environment
# and converting them into a state.


def _rgb_to_grayscale(image):
    """
    Convert an RGB-image into gray-scale using a formula from Wikipedia:
    https://en.wikipedia.org/wiki/Grayscale
    """

    # Get the separate colour-channels.
    r, g, b = image[:, :, 0], image[:, :, 1], image[:, :, 2]

    # Convert to gray-scale using the Wikipedia formula.
    img_gray = 0.2990 * r + 0.5870 * g + 0.1140 * b

    return img_gray


def _pre_process_image(image):
    # Pre-process a raw image from the game-environment.

    # Convert image to gray-scale.
    img = _rgb_to_grayscale(image)

    # Resize to the desired size using SciPy for convenience.
    img = scipy.misc.imresize(img, size=state_img_size, interp='bicubic')

    return img


class MotionTracer:
    """
    Used for processing raw image-frames from the game-environment.

    The image-frames are converted to gray-scale, resized, and then
    the background is removed using filtering of the image-frames
    so as to detect motions.

    This is needed because a single image-frame of the game environment
    is insufficient to determine the direction of moving objects.

    The original DeepMind implementation used the last 4 image-frames
    of the game-environment to allow the Neural network to learn how
    to detect motion. This implementation could make it a little easier
    for the Neural Network to learn how to detect motion, but it has
    only been tested on Breakout and Space Invaders, and may not work
    for games with more complicated graphics such as Doom. This remains
    to be tested.
    """

    def __init__(self, image, decay=0.75):
        """

        :param image:
            First image from the game-environment,
            used for resetting the motion detector.

        :param decay:
            Parameter for how long the tail should be on the motion-trace.
            This is a float between 0.0 and 1.0 where higher values means
            the trace / tail is longer.
        """

        # Pre-process the iamge and save it for later use.
        # The input iamge may be 8-bit integers but internally
        # we nned to use floating-point to avoid image-noise
        # caused by recurrent rounding-errors.
        img = _pre_process_image(image=image)
        self.last_input = img.astype(np.float)

        # Set the last output to zero.
        self.last_output = np.zeros_like(img)

        self.decay = decay

    def process(self, image):
        # Process a raw image-frame from the game-environment.

        # Pre-process the image so it is gray-scale and resized.
        img = _pre_process_image(image=image)

        # Subtract the previous input. This only leaves the
        # pixels that have changed in the two image-frames.
        img_dif = img - self.last_input

        # Copy the contents of the input-image to the last input.
        self.last_input[:] = img[:]

        # If the pixel-difference is greater than a threshold then
        # set the output pixel-value to the highest value (white),
        # otherwise set the output pixel-value to the lowest value (black).
        # So that we merely detect motion, and don't care about details.
        img_motion = np.where(np.abs(img_dif) > 20, 255.0, 0.0)

        # Add some of the previous output. This recurrent formula
        # is what gives the trace / tail.
        output = img_motion + self.decay * self.last_output

        # Ensure the pixel-values are within the allowed bounds.
        output = np.clip(output, 0.0, 255.0)

        # Set the last output.
        self.last_output = output

        return output

    def get_state(self):
        """
        Get a state that can be used as input to the Neural Network.

        It is basically just the last input and the last output of the
        motion-tracer. This means it is the last image-frame of the
        game-environment, as well as the motion-trace. This shows
        the current location of all the objects in the game-environment
        as well as trajectories / traces of where they have been.
        """

        # Stack the last input and output iamges.
        state = np.dstack([self.last_input, self.last_output])

        # Convert to 8-bit integer.
        # This is done to save space in the replay-memory.
        state = state.astype(np.uint8)

        return state

#######################################################################################


class ReplayMemory:
    """
    The replay-memory holds many previous states of the game-environment.
    This helps stabilize training of the Neural network because the data
    is more diverse when sampled over thousands of different states.
    """

    def __init__(self, size, num_actions, discount_factor=0.97):
        """

        :param size:
            Capacity of the replay-memory. This is the number of states.

        :param num_actions:
            Number of possible actions in the game-environment.

        :param discount_factor:
            Discount-factor used for updating Q-values.
        """

        # Array for the previous states of the game-environment.
        self.states = np.zeros(shape=[size] + state_shape, dtype=np.uint8)

        # Array for the Q-values corresponding to the states.
        self.q_values = np.zeros(shape=[size, num_actions], dtype=np.float)

        # Array for the Q-values before being updated.
        # This is used to compare the Q-values before and after the update.
        self.q_values_old = np.zeros(shape=[size, num_actions], dtype=np.float)

        # Actions taken for each of the states in the memory.
        self.actions = np.zeros(shape=size, dtype=np.int)

        # Reward observed for each of the states in the memory.
        self.rewards = np.zeros(shape=size, dtype=np.float)

        # Whether the life had ended in each state of the game-environment.
        self.end_life = np.zeros(shape=size, dtype=np.bool)

        # Whether the episode had ended (aka, game over) in each state.
        self.end_episode = np.zeros(shape=size, dtype=np.bool)

        # #stimation errors for the Q-values. This is used to balance
        # the sampling of batches for training the Neural Network,
        # so we get a balanced combination of states with high and low
        # estimation errors for their Q-values.
        self.estimation_erros = np.zeros(shape=size, dtype=np.float)

        # Capacity of the replay-memory as the number of states.
        self.size = size

        # Discount-factor for calculating Q-values.
        self.discount_factor = discount_factor

        # Reset the number of used states in the replay-memory.
        self.num_used = 0

        # Threshold for splitting between low and high estimation errors.
        self.error_threshold = 0.1

