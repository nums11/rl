import gymnasium as gym
from time import sleep
from tqdm import tqdm
import random
import numpy as np
from MCAgent import MCAgent
import sys

"""
0: Move left
1: Move down
2: Move right
3: Move up
"""
env = gym.make('FrozenLake-v1', desc=None, map_name="4x4", is_slippery=False)

original_stdout = sys.stdout

# First-Visit On-Policy MC Agent
file = open('MC_fv_on_policy_logs.txt', 'w')
sys.stdout = file
agent = MCAgent(env, first_visit=True, epsilon=0.5, gamma=0.9)
agent.trainOnPolicy(5000)
test_env = gym.make('FrozenLake-v1', desc=None, map_name="4x4", is_slippery=False,
	render_mode="human")
agent.test(test_env, 3)
file.close()

# Every-Visit On-Policy MC Agent
file = open('MC_ev_on_policy_logs.txt', 'w')
sys.stdout = file
agent = MCAgent(env, first_visit=False, epsilon=0.5, gamma=0.9)
agent.trainOnPolicy(1000)
test_env = gym.make('FrozenLake-v1', desc=None, map_name="4x4", is_slippery=False,
	render_mode="human")
agent.test(test_env, 3)
file.close()

