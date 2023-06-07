import gymnasium as gym
from time import sleep
from tqdm import tqdm
import random
import numpy as np


class MCAgent(object):
	"""docstring for MCAgent"""
	def __init__(self, env, on_policy=True, first_visit=True, epsilon=0.3, gamma=0.9):
		# Training Environment
		self.env = env
		# Sets on vs. off-policy learning
		self.on_policy = on_policy
		# Determines if the agent is using first visit or every-visit MC
		self.first_visit = first_visit
		# Degree of exploration
		self.epsilon = epsilon
		# Discount factor
		self.gamma = gamma
		self.num_states = self.env.observation_space.n
		self.num_actions = self.env.action_space.n
		self.Q = {}
		self.policy = {}
		# keeps track of the number of times each state-action pair is visited
		self.n = {}
		for s in range(self.num_states):
			self.policy[s] = 0
			for a in range(self.num_actions):
				self.Q[(s,a)] = 0
				self.n[(s,a)] = 0


	def train(self, num_episodes):
		if self.on_policy:
			self.trainOnPolicy(num_episodes)
		else:
			self.trainOffPolicy(num_episodes)

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
				# Store return and incrementally update action value function and policy 
				if (self.first_visit and self.isFirstVist(s_t, t, trajectory)) or (not self.first_visit):
					# Increment num times this state action pair was selected
					self.n[(s_t, a_t)] += 1
					# Update the action-value function
					self.Q[(s_t, a_t)] += (1 / self.n[(s_t,a_t)]) * (G - self.Q[(s_t, a_t)])
					# Make the policy greedy with respect to the action value function
					self.updatePolicy(s_t)
		print("Done Training-----------------------")
		print("Policy:", self.policy, '\n')
		print("Training Stats ---------------------")
		print(num_episodes, "episodes")
		print("Avg. cum_reward:", self.average(cum_rewards), '\n')

	def trainOffPolicy(self, num_episodes):
		print("Starting training-----------------")
		cum_rewards = []
		for episode in tqdm(range(num_episodes)):
			print("Episode", episode)
			# Generate an episode and receive the trajectory
			trajectory = self.genEpisode(cum_rewards)
			# Initialize G - discounted return
			G = 0
			# Importance Sampling Ratio
			W = 1
			# Iterate through trajectory in reverse
			for t in reversed(trajectory):
				print("T trajectory")
				# Grab state, action, and reward from this step of the trajectory
				s_t, a_t, r_t_plus_1 = trajectory[t]
				# Add to the discounted return
				G = self.gamma * G + r_t_plus_1
				# Store return and incrementally update action value function and policy 
				if (self.first_visit and self.isFirstVist(s_t, t, trajectory)) or (not self.first_visit):
					# Increment num times this state action pair was selected
					self.n[(s_t, a_t)] += 1
					# Update the action-value function
					# How should I update G?
					self.Q[(s_t, a_t)] += (1 / self.n[(s_t,a_t)]) * (G - self.Q[(s_t, a_t)])
					# Make the policy greedy with respect to the action value function
					self.updatePolicy(s_t)
					# Update the importance sampling ratio
					# If this action wouldn't be selected in the target then
					# move onto next episode as the rest of the trajectory is not possible
					if a_t != self.policy[s_t]:
						print("Breaking.should exit episode")
						break
					# target policy is det. so likelihood of it being selected is 1
					# behavior policy is random so likelihood of it being 1 / num actions
					W *= 1 / (1 / len(self.num_actions))

		print("Done Training-----------------------")
		print("Policy:", self.policy, '\n')
		print("Training Stats ---------------------")
		print(num_episodes, "episodes")
		print("Avg. cum_reward:", self.average(cum_rewards), '\n')

	def genEpisode(self, cum_rewards):
		obs, transition_prob = self.env.reset()
		terminated = False

		trajectory = {}
		t = 0
		cum_reward = 0
		while not terminated:
			s = obs
			action = self.getPolicyAction(s)
			# print("Taking action", action)
			obs, reward, terminated, truncated, transition_prob = self.env.step(action)
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

	# For an on-policy agent: returns actions following an epsilon greedy policy
	# For an off-policy agent: returns an action following behavior policy (random policy)
	def getPolicyAction(self, s):
		if self.on_policy:
			if random.uniform(0,1) < self.epsilon:
				return random.choice(list(range(self.num_actions)))
			else:
				return self.policy[s]
		else:
			return random.choice(list(range(self.num_actions)))

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

	# Update policy to choose the max action from the given state
	# based on the action value function
	def updatePolicyState(self, s):
		max_a = self.getMaxActionForState(s)
		self.policy[s] = max_a

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
			obs, transition_prob = self.env.reset()
			print("Episode", episode)
			terminated = False
			cum_reward = 0
			while not terminated:
				action = self.policy[obs]
				print("Taking action", action)
				obs, reward, terminated, truncated, info = self.env.step(action)
				if reward > 0:
					print("reached goal!")
				cum_reward += reward
				if terminated:
					print("Episode", episode, "finished with cum_reward", cum_reward)
					cum_rewards.append(cum_reward)
		print("\nDone Testing. Stats -------------------")
		print(num_episodes, "episodes")
		print("Avg. cum_reward:", self.average(cum_rewards))