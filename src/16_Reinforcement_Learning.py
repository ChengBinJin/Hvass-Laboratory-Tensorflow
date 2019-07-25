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
