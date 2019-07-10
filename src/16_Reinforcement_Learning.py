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
