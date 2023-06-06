import gymnasium as gym
from time import sleep
from tqdm import tqdm
import random
import numpy as np

"""
0: Move left
1: Move down
2: Move right
3: Move up
"""
env = gym.make('FrozenLake-v1', desc=None, map_name="4x4", is_slippery=False)

# stuff = env.reset()
# print(stuff)
# sleep(2)
# observation, reward, terminated, truncated, info = env.step(1)

# print(observation)
# print(reward)
# print(terminated)
# print(truncated)
# print(info)


# env.render()
# sleep(2)


class MCAgent(object):
	"""docstring for MCAgent"""
	def __init__(self, env, epsilon=0.3, gamma=0.9):
		self.env = env
		self.epsilon = epsilon
		self.gamma = gamma
		self.num_states = env.observation_space.n
		self.num_actions = env.action_space.n
		self.Q = {}
		self.policy = {}
		self.returns = {}
		for s in range(self.num_states):
			self.policy[s] = 0
			for a in range(self.num_actions):
				self.Q[(s,a)] = 0
				self.returns[(s,a)] = []

	# Trains using the on policy first visit MC Control Algorithm
	def trainOnPolicy(self, num_episodes):
		print("Starting training-----------------")
		cum_rewards = []
		for episode in tqdm(range(num_episodes)):
			print("Episode", episode)
			# Generate an episode and receive the trajectory
			trajectory = self.genEpisode(cum_rewards)
			# Initialize G - discounted return
			G = 0
			# Iterate through trajectory in reverse
			for t in reversed(trajectory):
				# Grab state, action, and reward from this step of the trajectory
				s_t, a_t, r_t_plus_1 = trajectory[t]
				# Add to the discounted return
				G = self.gamma * G + r_t_plus_1
				# Store return and update action value function and policy if this is the first
				# visit of this state
				if self.isFirstVist(s_t, t, trajectory):
					self.returns[(s_t, a_t)].append(G)
					self.Q[(s_t, a_t)] = self.average(self.returns[(s_t, a_t)])
					self.updatePolicyState(s_t, self.getMaxActionForState(s_t))
		print("Done Training-----------------------")
		print("Policy:", self.policy, '\n')
		print("Training Stats ---------------------")
		print(num_episodes, "episodes")
		print("Avg. cum_reward:", self.average(cum_rewards), '\n')

	def genEpisode(self, cum_rewards):
		obs, transition_prob = env.reset()
		terminated = False

		trajectory = {}
		t = 0
		cum_reward = 0
		while not terminated:
			s = obs
			action = self.getPolicyAction(s)
			# print("Taking action", action)
			obs, reward, terminated, truncated, transition_prob = env.step(action)
			cum_reward += reward
			trajectory[t] = [s,action,reward]
			t += 1
			if reward > 0:
				print("reached goal!")
			if terminated:
				print("Episode Terminated. cum_reward", cum_reward, "truncated?", truncated,
					"t:", t, "final_obs:", obs)
				cum_rewards.append(cum_reward)
		return trajectory

	# Returns actions following an epsilon greedy policy
	def getPolicyAction(self, s):
		if random.uniform(0,1) < self.epsilon:
			return random.choice(list(range(self.num_actions)))
		else:
			return self.policy[s]

	# Returns if this is the first visit of a state in a trajectory
	def isFirstVist(self, s_t, t, trajectory):
		for timestep in trajectory:
			s,a,r = trajectory[timestep]
			if s == s_t:
				return timestep == t

	# Gets the average value from a licst
	def average(self, lst):
		if len(lst) == 0:
			return 0
		return sum(lst) / len(lst)

	# Update policy to choose action 'a' when in state 's'
	def updatePolicyState(self, s, a):
		self.policy[s] = a

	# Gets the action that has the highest Q value for this state
	def getMaxActionForState(self, s):
		max_value = float('-inf')
		max_action = None
		for (state, action), value in self.Q.items():
			if state == s:
				if value > max_value:
					max_action = action
					max_value = value
		return max_action

	# Test the agent on the test environment for the number of
	# specified episodes following a determininstic policy
	def test(self, env, num_episodes):
		print("Testing-----------------------------")
		cum_rewards = []
		for episode in range(num_episodes):
			obs, transition_prob = env.reset()
			print("Episode", episode)
			terminated = False
			cum_reward = 0
			while not terminated:
				action = self.policy[obs]
				print("Taking action", action)
				obs, reward, terminated, truncated, info = env.step(action)
				if reward > 0:
					print("reached goal!")
				cum_reward += reward
				if terminated:
					print("Episode", episode, "finished with cum_reward", cum_reward)
					cum_rewards.append(cum_reward)
		print("Done Testing. Stats -------------------")
		print(num_episodes, "episodes")
		print("Avg. cum_reward:", self.average(cum_rewards))



agent = MCAgent(env, epsilon=0.5, gamma=0.9)
agent.trainOnPolicy(5000)

test_env = gym.make('FrozenLake-v1', desc=None, map_name="4x4", is_slippery=False,
	render_mode="human")

agent.test(test_env, 3)


# def OnPolicyFirstVistMCControl(gamma, num_episodes, epsilon=0.1):
# 	# Initialize e-greedy policy with initial actions chosen
# 	# randomly for each state
# 	policy = Policy('eg', epsilon)

# 	# Initialize Q (action-value function) and returns for the 4x4 grid world
# 	Q = {}
# 	returns = {}
# 	for x in range(4):
# 		for y in range(4):
# 			for action in range(4):
# 				Q[((x,y), action)] = 0 # Initial action values are 0
# 				returns[((x,y), action)] = []

# 	# loop number of episodes
# 	for _ in tqdm(range(num_episodes)):
# 		# Generate an episode and receive the trajectory
# 		trajectory = genEpisode(env, policy)
# 		# Initialize G - discounted return
# 		G = 0
# 		# Iterate through trajectory in reverse
# 		for t in reversed(trajectory):
# 			# Grab state, action, and reward from this step of the trajectory
# 			s_t, a_t, r_t_plus_1 = trajectory[t]
# 			# Add to the discounted return
# 			G = gamma * G + r_t_plus_1
# 			# Store return and update action value function and policy if this is the first
# 			# visit of this state
# 			if isFirstVist(s_t, t, trajectory):
# 				returns[(s_t, a_t)].append(G)
# 				Q[(s_t, a_t)] = average(returns[(s_t, a_t)])
# 				policy.updateDetPolicyState(s_t, getMaxActionForState(s_t, Q))

# 	printPolicy(policy)
