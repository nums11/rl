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
# file = open('MC_fv_on_policy_logs.txt', 'w')
# sys.stdout = file
# agent = MCAgent(env, first_visit=True, epsilon=0.5, gamma=0.9)
# agent.trainOnPolicy(5000)
# test_env = gym.make('FrozenLake-v1', desc=None, map_name="4x4", is_slippery=False,
# 	render_mode="human")
# agent.test(test_env, 3)
# file.close()

# Every-Visit On-Policy MC Agent
# file = open('MC_ev_on_policy_logs.txt', 'w')
# sys.stdout = file
# agent = MCAgent(env, first_visit=False, epsilon=0.5, gamma=0.9)
# agent.trainOnPolicy(1000)
# test_env = gym.make('FrozenLake-v1', desc=None, map_name="4x4", is_slippery=False,
# 	render_mode="human")
# agent.test(test_env, 3)
# file.close()

# First-Visit Off-Policy Ordinary Importance Sampling MC Agent
# file = open('MC_fv_off_policy_ord_is_logs.txt', 'w')
# sys.stdout = file
# agent = MCAgent(env, on_policy=False, first_visit=True, gamma=0.9)
# agent.train(5000)
# test_env = gym.make('FrozenLake-v1', desc=None, map_name="4x4", is_slippery=False,
# 	render_mode="human")
# agent.test(test_env, 3)
# file.close()

# Every-Visit Off-Policy Ordinary Importance Sampling MC Agent
# file = open('MC_ev_off_policy_ord_is_logs.txt', 'w')
# sys.stdout = file
# agent = MCAgent(env, on_policy=False, first_visit=False, gamma=0.9)
# agent.train(10000)
# test_env = gym.make('FrozenLake-v1', desc=None, map_name="4x4", is_slippery=False,
# 	render_mode="human")
# agent.test(test_env, 3)
# file.close()

# First-Visit Off-Policy Weighted Importance Sampling MC Agent
# file = open('MC_fv_off_policy_weighted_is_logs.txt', 'w')
# sys.stdout = file
# agent = MCAgent(env, on_policy=False, weighted_is=True, first_visit=True, gamma=0.9)
# agent.train(5000)
# test_env = gym.make('FrozenLake-v1', desc=None, map_name="4x4", is_slippery=False,
# 	render_mode="human")
# agent.test(test_env, 3)
# file.close()

# Every-Visit Off-Policy Ordinary Importance Sampling MC Agent
# file = open('MC_ev_off_policy_weighted_is_logs.txt', 'w')
# sys.stdout = file
# agent = MCAgent(env, on_policy=False, weighted_is=True, first_visit=False, gamma=0.9)
# agent.train(5000)
# test_env = gym.make('FrozenLake-v1', desc=None, map_name="4x4", is_slippery=False,
# 	render_mode="human")
# agent.test(test_env, 3)
# file.close()