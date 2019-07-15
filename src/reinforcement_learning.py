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
        self.estimation_errors = np.zeros(shape=size, dtype=np.float)

        # Capacity of the replay-memory as the number of states.
        self.size = size

        # Discount-factor for calculating Q-values.
        self.discount_factor = discount_factor

        # Reset the number of used states in the replay-memory.
        self.num_used = 0

        # Threshold for splitting between low and high estimation errors.
        self.error_threshold = 0.1

    def is_full(self):
        # Return boolean whether the replay-memory is full.
        return self.num_used == self.size

    def used_fraction(self):
        # Return the fraction of the replay-memory that is used.
        return self.num_used / self.size

    def reset(self):
        # Reset the replay-memory so it is empty.
        self.num_used = 0

    def add(self, state, q_values, action, reward, end_life, end_episode):
        """
        Add an observed state from the game-environment, along with the
        estimated Q-values, action taken, observed reward, etc.

        :param state:
            Current state of the game-environment.
            This is the output of the MotionTracer-class.

        :param q_values:
            The estimated Q-values for the state.

        :param action:
            The action taken by the agent in this state of the game.

        :param reward:
            The reward that was observed from taking this action
            and moving to the next state.

        :param end_life:
            Boolean whether the agent has lost a life in this state.

        :param end_episode:
            Boolean whether the agent has lost all lives aka. game over
            aks. end of episode.
        """

        if not self.is_full():
            # Index into the arrays for convenience.
            k = self.num_used

            # Increase the number of used elemetns in the replay-memory.
            self.num_used += 1

            # Store all the values in the replay-memory.
            self.states[k] = state
            self.q_values[k] = q_values
            self.actions[k] = action
            self.end_life[k] = end_life
            self.end_episode[k] = end_episode

            # Note that the reward is limited. This is done to stabilize
            # the trainign of the Neural Network.
            self.rewards[k] = np.clip(reward, -1.0, 1.0)

    def update_all_q_values(self):
        """
        Update all Q-values in the replay-memory.

        When states and Q-values are added to the replay-memory, the
        Q-values have been estimated by the Neural Network. But we now
        have more data available that we can use to improve the estimated
        Q-values, because we not know which actions were taken and the
        observed rewards. We sweep backwards through the entire replay-memory
        to use the observed data to improve the estimated Q-values.
        """

        # Copy old Q-values so we can print their statiscs later.
        # Note that the contents of the arrays are copied.
        self.q_values_old[:] = self.q_values[:]

        # Process the replay-memory backwards and update the Q-values.
        # This looop could be implemented entirely in Numpy for higher speed,
        # but it is probably only a small fraction of the overall time usage,
        # and it is much easier to understand when implemented like this.
        for k in reversed(range(self.num_used-1)):
            # Get the data for the k'th state in the replay-memory.
            action = self.actions[k]
            reward = self.rewards[k]
            end_life = self.end_life[k]
            end_episode = self.end_episode[k]

            # Calculate the Q-value for the action that was taken in this state.
            if end_life or end_episode:
                # If the agent lost a life or it was game over / end of episode,
                # then the value of taking the given action is just the reward
                # that was observed in this single step. This is because the
                # Q-value is defined as the discounted value of all future game
                # steps in a single life of the agent. When the life has ended,
                # there will be no future steps.
                action_value = reward
            else:
                # Otherwise the value of taking the action is the reward that
                # we have observed plus the discounted value of future rewards
                # from continuing the game. We use the estimated Q-values for
                # the following state and take the maximum, because we will
                # generally take the action that has the highest Q-value.
                action_value = reward + self.discount_factor * np.max(self.q_values[k + 1])

            # Error of the Q-value that was estimated using the Neural Network.
            self.estimation_errors[k] = abs(action_value - self.q_values[k, action])

            # Update the Q-value wit hthe better estimate.
            self.q_values[k, action] = action_value

        self.print_statistics()

    def prepare_sampling_prob(self, batch_size=128):
        """
        Prepare the probability distribution for random sampling of stats
        and Q-values for use in training of the Neural Network.

        The probability distribution is just a simple binary split of the
        replay-memory based on the estimation errors of the Q-values.
        The idea is to create a batch of samples that are balanced somewhat
        evenly between Q-values that the Neural Network already knows how to
        estimate quiite well because they have low estimation errors, and
        Q-values that are poorly estimated by the Neural network because
        they have high estiamtinon errors.

        The reason for this balancing of Q-values with high and low estimation
        errors, is that if we train the Neural network mostly on data with
        high estimation errors, then it will tend to forget what it already
        knows and hence become over-fit so the training becomes unstable.
        """

        # Get the errors between the Q-values that were estimated using
        # the Neural Network, and the Q-values that were updated with the
        # reward that was actually observed when an action was taken.
        err = self.estimation_errors[0:self.num_used]

        # Create an index of the estimation errors that are low.
        idx = err < self.error_threshold
        self.idx_err_lo = np.squeeze(np.where(idx))

        # Create an index of the estimation errors that are high.
        self.idx_err_hi = np.squeeze(np.where(np.logical_not(idx)))

        # Probability of sampling Q-values with high estiamtion errors.
        # This is either set to the fraction of the replay-memory that
        # has high estimation errors - or it is set to 0.5. So at least
        # half of the batch has high estimation errors.
        prob_err_hi = len(self.idx_err_hi) / self.num_used
        prob_err_hi = max(prob_err_hi, 0.5)

        # Number of samples in a batch that have high estimation errors.
        self.num_samples_err_hi = int(prob_err_hi * batch_size)

        # Number of samples in a batch that have low estimation errors.
        self.num_samples_err_lo = batch_size - self.num_samples_err_hi

    def random_batch(self):
        """
        Get a random batch of states and Q-values from the replay-memory.
        You must call prepare_sampling_prob() before calling this function,
        which also sets the batch-size.

        The batch has been balanced so it contains states and Q-values
        that have both high and low estimation errors for the Q-values.
        This is done to both speed up and stabilize training of the
        Neural Network.
        """

        # Random index of states and Q-values in the replay-memory.
        # These have LOW estimation errors for the Q-values.
        idx_lo = np.random.choice(self.idx_err_lo,
                                  size=self.num_samples_err_lo,
                                  replace=False)

        # Random index of states and Q-values in the replay-memory.
        # These have HIGH estimation errors for the Q-values.
        idx_hi = np.random.choice(self.idx_err_hi,
                                  size=self.num_samples_err_hi,
                                  replace=False)

        # Combine the indices.
        idx = np.concatenate((idx_lo, idx_hi))

        # Get the batches of states and Q-values.
        states_batch = self.states[idx]
        q_values_batch = self.q_values[idx]

        return states_batch, q_values_batch

    def all_batches(self, batch_size=128):
        """
        Iterator for all states and Q-values in the replay-memory.
        It returns the indices for the beginning and end, as well as
        a progress-counter between 0.0 and 1.0.

        This function is not currently being used except by the function
        estimate_all_q_values() below. These two functions are merely
        included to make it easier for you to experiment with the code
        by showing you an easy and efficient way to loop over all the
        data in the replay-memory.
        """

        # Start index for the current batch.
        begin = 0

        # Repeat untill all batches have been processed.
        while begin < self.num_used:
            # End index for the current batch.
            end = begin + batch_size

            # Ensure the batch does not exceed the used replay-memory.
            if end > self.num_used:
                end = self.num_used

            # Progress counter.
            progress = end / self.num_used

            # Yield the batch indices and completion-counter.
            yield begin, end, progress

            # Set the start-index for the next batch to the end of this batch.
            begin = end

    def estimate_all_q_values(self, model):
        """
        Estimate all Q-values for the states in the replay-memory
        using the model / Neural Network.

        Note that this function is not currently being used. It is provided
        to make it easier for you to experiment with this code, by showing
        you an efficient way to iterate over all the states and Q-values.

        :param model:
            Instance of the NeuralNetwork-class.
        """

        print("Re-calculating all Q-values in replay memory ...")

        # Process the entire replay-memory in batches.
        for begin, end, progress in self.all_batches():
            # Print progress.
            msg = "\tProgress: {0:.0%}"
            msg = msg.format(progress)
            print_progress(msg)

            # Get the states for the current batch.
            states = self.states[begin:end]

            # Calculate the Q-values using the Neural Network
            # and update the replay-memory.
            self.q_values[begin:end] = model.get_q_values(states=states)

        # Newline.
        print()


