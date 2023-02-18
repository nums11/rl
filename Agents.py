import random
import numpy as np

class Action(object):
	"""
	Implementation of an action.

	optimistic: determines whether or not Q is initialized optimistically
	"""
	def __init__(self, optimistic=False):
		# Current value estimation for this action
		self.Q = 10 if optimistic else 0
		# number of times this action has been called
		self.n = 0
		
class MultiArmedBanditsAgent(object):
	"""
	Base implementation of an Agent for the Multi-Armed Bandits Problem

	env: Gym env the agent will be trained on
	"""
	def __init__(self, env):
		# Create list of actions agent can select from env (e.g. [0,1])
		self.actions = []
		for _ in range(env.action_space.n):
			self.actions.append(Action(optimistic=True))

	def selectAction(self):
		pass

	# Selects an action greedily - chooses action with highest Q	
	def selectActionGreedy(self):
		best_action = 0
		highest_q = self.actions[0].Q

		for i in range(1, len(self.actions)):
			curr_action_Q = self.actions[i].Q
			if curr_action_Q > highest_q:
				highest_q = curr_action_Q
				best_action = i

		return best_action

	# Selects an action epsilon greedily
	# Chooses action greedily (1-epsilon) % of the time
	# and explores a random action epsilon percent of the time	
	def selectActionEpsilonGreedy(self, epsilon):
		if random.uniform(0,1) < epsilon:
			# Explore
			return random.choice(list(range(len(self.actions))))
		else:
			# Exploit
			return self.selectActionGreedy()

	# Updates estimated value 'Q' of an action given the reward
	# that was just given for taking the action using the incremental
	# update algorithm
	def updateActionEstimation(self, action, reward, alpha=None):
		self.actions[action].n += 1

		step_size = (1 / self.actions[action].n)
		if alpha != None:
			step_size = alpha

		self.actions[action].Q = self.actions[action].Q + \
			step_size * (reward - self.actions[action].Q)


class GreedyAgent(MultiArmedBanditsAgent):
	"""
	Implementation of a greedy agent.

	env: Gym env the agent will be trained on
	"""
	def __init__(self, env):
		# Call base class constructor
		super().__init__(env)

	def selectAction(self):
		return super().selectActionGreedy()
		

class EpsilonGreedyAgent(MultiArmedBanditsAgent):
	"""
	Implementation of an epsilon greedy agent.

	env: Gym env the agent will be trained on

	epsilon: epsilon-greedy exploration parameter
	"""
	def __init__(self, env, epsilon):
		# Call base class constructor
		super().__init__(env)
		self.epsilon = epsilon

	def selectAction(self):
		return super().selectActionEpsilonGreedy(self.epsilon)

class ConstantAlphaAgent(MultiArmedBanditsAgent):
	"""
	Implementation of an agent that uses a constant alpha value
	for action-value estimation and epsilon-greedy action selection.

	env: Gym env the agent will be trained on

	epsilon: epsilon-greedy exploration parameter

	alpha: step size
	"""
	def __init__(self, env, epsilon, alpha):
		# Call base class constructor
		super().__init__(env)
		self.epsilon = epsilon
		self.alpha = alpha

	def selectAction(self):
		return super().selectActionEpsilonGreedy(self.epsilon)

	def updateActionEstimation(self, action, reward):
		return super().updateActionEstimation(action, reward, self.alpha)

class UCBAgent(MultiArmedBanditsAgent):
	"""
	Implementation of an agent that uses a constant alpha value
	for action-value estimation and Upper-Confidence-Bound action
	selection.

	env: Gym env the agent will be trained on

	alpha: step size

	c: confidence value
	"""
	def __init__(self, env, alpha, c):
		# Call base class constructor
		super().__init__(env)
		self.alpha = alpha
		self.c = c

	# Selects an action using the UCB algorithm
	def selectAction(self, t):
		ucb_values = []
		for action in self.actions:
			uncertainty = None
			if action.n == 0:
				uncertainty = float("inf")
			else:
				uncertainty = self.c * np.sqrt(np.log(t) / action.n)

			# UCB formula is estimation + uncertainty
			ucb_values.append(
				action.Q + uncertainty
			)

		return np.argmax(ucb_values)


	def updateActionEstimation(self, action, reward):
		return super().updateActionEstimation(action, reward, self.alpha)
