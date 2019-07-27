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

# Output of Convolutional Layer

# Game State



# Output of Convolutional Layer 1

# Output of Convolutional Layer 2

# Output of Convolutional Layer 3



# Weights for Convolutional Layers

# Weights for Convolutional Layer 1

# Weights for Convolutional Layer 2
