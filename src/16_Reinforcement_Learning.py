# Imports
import math
import gym
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

import reinforcement_learning as rl

print("TensorFlow version: {}".format(tf.__version__))  # TensorFlow
print("Gym version: {}".format(gym.__version__))  # OpenAI Gym

# Game Environment
env_name = "Breakout-v0"
# env_name = "SpaceInvaders-v0"

rl.checkpoint_base_dir = '../checkpoints_tutorial16/'
rl.update_paths(env_name=env_name)

# Create Agent
agent = rl.Agent(env_name=env_name,
                 training=True,
                 render=True,
                 use_logging=False)

model = agent.model

replay_memory = agent.replay_memory

# Training
agent.run(num_episodes=1)

# Training Progress
log_q_values = rl.LogQValues()
log_reward = rl.LogReward()

log_q_values.read()
log_reward.read()

# Training Progress: Reward
plt.plot(log_reward.count_states, log_reward.episode, label="Episode Reward")
plt.plot(log_reward.count_states, log_reward.mean, label="Mean of 30 episodes")
plt.xlabel("State-Count for Game Environment")
plt.legend()
plt.show()

# Training Progress: Q-Values
plt.plot(log_q_values.count_states, log_q_values.mean, label='Q-Value Mean')
plt.xlabel('State-Count for Game Environment')
plt.legend()
plt.show()

# Testing
print("agent.epsilon_greedy.epsilon_testing: {}".format(agent.epsilon_greedy.epsilon_testing))

agent.training = False
agent.render = True
agent.run(num_episodes=1)

# Mean Reward
agent.reset_episode_rewards()
agent.render = False
agent.run(num_episodes=30)

rewards = agent.episode_rewards
print("Rewards for {0} episodes:".format(len(rewards)))
print("- Min:   ", np.min(rewards))
print("- Mean:  ", np.mean(rewards))
print("- Max:   ", np.max(rewards))
print("- Stdev: ", np.std(rewards))

_ = plt.hist(rewards, bins=30)

# Example States
def print_q_values(idx):
    """Print Q-values and actions from the replay-memory at the given index."""

    # Get the Q-values and action from the replay-memory.
    q_values = replay_memory.q_values[idx]
    action = replay_memory.actions[idx]

    print("Action:      Q-Value:")
    print("=====================")

    # Print all the actions and their Q-values.
    for i, q_value in enumerate(q_values):
        # Used to display which action was taken.
        if i == action:
            action_taken = "(Action Taken)"
        else:
            action_taken = ""

        # Text-name of the action.
        action_name = agent.get_action_name(i)

        print("{0:12}{1:3f} {2}".format(action_name, q_value, action_taken))

    # Newline.
    print()

def plot_state(idx, print_q=True):
    """Plot the state in the replay-memory with the given index."""

    # Get the state from the replay-memory.
    state = replay_memory.states[idx]

    # Create figure with a grid of sub-plots.
    fig, axes = plt.subplots(1, 2)

    # Plot the image from the game-environment.
    ax = axes.flat[0]
    ax.imshow(state[:, :, 0], vmin=0, vmax=255, interpolatin='lanczos', cmap='gray')

    # Plot the motion-trace.
    ax = axes.flat[1]
    ax.imshow(state[:, :, 1], vmin=0, vmax=255, interpolation='lanczos', cmap='gray')

    # This is necessary if we show more than one plot in a single Notebook cell.
    plt.show()

    # Print the Q-values.
    if print_q:
        print_q_values(idx=idx)

num_used = replay_memory.num_used
print('num_used: {}'.format(num_used))

q_values = replay_memory.q_values[0:num_used, :]
q_values_min = q_values.min(axis=1)
q_values_max = q_values.max(axis=1)
q_values_dif = q_values_max - q_values_min

# Example States: Highest Reward
idx = np.argmax(replay_memory.rewards)
print("Example States: Hightest Reward")
print('idx: {}'.format(idx))

for i in range(-5, 3):
    plot_state(idx=idx+i)


# Example: Highest Q-value
idx = np.argmax(q_values_max)
print("Example: Highest Q-Value")
print('idx: {}'.format(idx))

for i in range(0, 5):
    plot_state(idx=idx+i)

# Example: Loss of life
idx = np.argmax(replay_memory.end_life)
print("Example: Loss of Life")
print("idx: {}".format(idx))

for i in range(-10, 0):
    plot_state(idx=idx+i)

# Example: Difference in Q-values
idx = np.argmax(q_values_dif)
print("Example: Greatest Difference in Q-Values")
print("idx: {}".format(idx))

for i in range(0, 5):
    plot_state(idx=idx+i)

# Example: Smallest Difference in Q-Values
idx = np.argmin(q_values_dif)
print("Example: Smallest Difference in Q-Values")
print("idx: {}".format(idx))

for i in range(0, 5):
    plot_state(idx=idx+i)

# Output of Convolutional Layer
def plot_layer_output(model, layer_name, state_index, inverse_cmap=False):
    """
    Plot the output of aconvolutional layer.

    :param model: An instance of the NeuralNetwork-class.
    :param layer_name: Name of the convolutional layer.
    :param state_index: Index into the replay-memory for a state taht
                        will be input to the Neural Network.
    :param inverse_cmap: Boolean whether to inverse the color-map.
    """

    # Get the given state-array from the replay-memory.
    state = replay_memory.states[state_index]

    # Get the output tensor for the given layer insdie the TensorFlow graph.
    # This is not the value-contents but merely a reference to the tensor.
    layer_tensor = model.get_layer_tensor(layer_name=layer_name)

    # Get the actual value of the tensor by feeding the state-data
    # to the TensorFlow graph and calculating the value of the tensor.
    values = model.get_tensor_value(tensor=layer_tensor, state=state)

    # Number of image channels output by the convolutional layer.
    num_images = values.shape[3]

    # Number of grid-cells to plot.
    # Rounded-up, square-root of the number of filters.
    num_grids = math.ceil(math.sqrt(num_images))

    # Create figure with a grid of sub-plots.
    fig, axes = plt.subplots(num_grids, num_grids, figsize=(10, 10))

    print("Dim. of each image:", values.shape)

    if inverse_cmap:
        cmap = 'gray_r'
    else:
        cmap = 'gray'

    # Plot the outputs of all the channels in the conv-layer.
    for i, ax in enumerate(axes.flat):
        # Only plot the valid iamge-channels.
        if i < num_images:
            # Get the image for the i'th output channel.
            img = values[0, :, :, i]

            # Plot image.
            ax.imshow(img, interpolation='nearest', cmap=cmap)

        # Remove ticks from the plot.
        ax.set_xticks([])
        ax.set_yticks([])

    # Ensure the plot is shown correctly with multiple plots
    # in a single Notebook cell.
    plt.show()

# Game State
idx = np.argmax(q_values_max)
plot_state(idx=idx, print_q=False)

# Output of Convolutional Layer 1
plot_layer_output(model=model, layer_name='layer_conv1', state_index=idx, inverse_cmap=False)

# Output of Convolutional Layer 2
plot_layer_output(model=model, layer_name='layer_conv2', state_index=idx, inverse_cmap=False)

# Output of Convolutional Layer 3
plot_layer_output(model=model, layer_name='layer_conv3', state_index=idx, inverse_cmap=False)

# Weights for Convolutional Layers
def plot_conv_weights(model, layer_name, input_channel=0):
    """
    Plot the weights for a convolutional layer.

    :param model: An instance of the NeuralNetwork-class.
    :param layer_name: Name of the convolutional layer.
    :param input_channel: Plot the weights for this input-channel.
    """

    # Get the variable for the weights of the given layer.
    # This is a reference to the variable inside TensorFlow,
    # not its actual value.
    weights_variable = model.get_weights_variable(layer_name=layer_name)

    # Retrieve the values of the weight-variable from TensorFlow.
    # The format of this 4-dim tensor is determined by the
    # TensorFlow API. See Tutorial #02 for more details.
    w = model.get_variable_value(variable=weights_variable)

    # Get the weights for the given input-channel.
    w_channel = w[:, :, input_channel, :]

    # Number of output-channels for the conv. layer.
    num_output_channels = w_channel.shape[2]

    # Get the lowest and highest values for the weights.
    # This is used to correct the colour intensity across
    # the images so they can compared with each other.
    w_min = np.min(w_channel)
    w_max = np.max(w_channel)

    # This is used to center the colour intensity as zero.
    abs_max = max(abs(w_min), abs(w_max))

    # Print statistics for the weights.
    print("Min: {0:.5f}, Max: {1:.5f}".format(w_min, w_max))
    print("Mean: {0:.5f}, Stdev: {1:.5f}".format(w_channel.mean(), w_channel.std()))

    # Number of grids to plot.
    # Rounded-up, square-root of the number of output-channels.
    num_grids = math.ceil(math.sqrt(num_output_channels))

    # Create figure with a grid of sub-plots.
    fig, axes = plt.subplots(num_grids, num_grids)

    # Plot all the filter-weights.
    for i, ax in enumerate(axes.flat):
        # Only plot the valid filter-weights.
        if i < num_output_channels:
            # Get the weights for the i'th filter of this input-channel.
            img = w_channel[:, :, i]

            # Plot image.
            ax.imshow(img, vmin=-abs_max, vmax=abs_max, interpolation='nearest', cmap='seismic')

        # Remove ticks from the plot.
        ax.set_xticks([])
        ax.set_yticks([])

    # Ensure the plot is shown correctly with multiple plots.
    # in a single Notebook cell.
    plt.show()

# Weights for Convolutional Layer 1
plot_conv_weights(model=model, layer_name='layer_conv1', input_channel=0)
plot_conv_weights(model=model, layer_name='layer_conv1', input_channel=1)

# Weights for Convolutional Layer 2
plot_conv_weights(model=model, layer_name='layer_conv2', input_channel=0)

# Weights for Convolutional Layer 3
plot_conv_weights(model=model, layer_name='layer_conv3', input_channel=0)

