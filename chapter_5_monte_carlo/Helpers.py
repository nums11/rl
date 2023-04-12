import numpy as np
import random
import pprint

class Policy(object):
	"""
	Represents a policy

	p_type: type of policy
	- d: deterministic
	- nd: non-deterministic
	- eg: epsilon-greedy

	epsilon: optional epsilon parameter for eg policy.
	"""
	def __init__(self, p_type, epsilon=0.1):
		valid_policy_types = ['d', 'nd', 'eg']
		assert(p_type in valid_policy_types)
		self.p_type = p_type
		# Actual policy is stored as a python dictionary
		self.policy = {}
		self.epsilon = epsilon
		self.initPolicyValues()

	# Initializes policy values based on the type of policy
	def initPolicyValues(self):
		for x in range(4):
			for y in range(4):
				if self.p_type == 'nd':
					# Non-deterministic policies are initialized with
					# all actions having equal probability of being selected
					self.policy[(x,y)] = [0.25] * 4
				else:
					# Deterministic and E-Greedy policies are initialized
					# with actions chosen randomly for each state
					self.policy[(x,y)] = random.randint(0,3)

	# Return the action the policy should take given the state 's'
	def getPolicyAction(self, s):
		if self.p_type == 'd':
			return self.policy[s]
		elif self.p_type == 'nd':
			return self._getNonDetPolicyAction(s)
		elif self.p_type == 'eg':
			return self._getEpGreedyPolicyAction(s)

	# Samples an action from the policy with the distribution as the likelihood
	# of each action being chosen
	def _getNonDetPolicyAction(self, s):
		return np.random.choice(
			np.arange(0,4),
			p=self.policy[s])

	# Gets an action greedily with prob. 1 - epsilon or randomly with
	# prob. epsilon
	def _getEpGreedyPolicyAction(self, s):
		if random.uniform(0,1) < self.epsilon:
			# Explore
			return random.choice(list(range(4)))
		else:
			# Exploit
			return self.policy[s]

	# Update policy to choose action 'a' when in state 's'
	def updateDetPolicyState(self, s, a):
		self.policy[s] = a

# Gets the average value from a licst
def average(lst):
	if len(lst) == 0:
		return 0
	return sum(lst) / len(lst)

# Returns if this is the first visit of a state in a trajectory
def isFirstVist(s_t, t, trajectory):
	for timestep in trajectory:
		s,a,r = trajectory[timestep]
		if s == s_t:
			return timestep == t

# Generates an episode
# es: optionally use exploring starts
def genEpisode(env, policy, es=False):
	obs = env.reset()
	done = False

	# Generate episode with exploring starts
	if es:
		# Choose a random start state and action
		x = random.randint(0,3)
		y = random.randint(0,3)
		a = random.randint(0,3)
		env.setStartPos((x,y))
		obs, reward, done, info = env.step(a)

	trajectory = {}
	t = 0
	while not done:
		s = obs
		action = policy.getPolicyAction(s)
		obs, reward, done, info = env.step(action)
		trajectory[t] = [s,action,reward]
		t += 1
	return trajectory

# Prints state for the grid world
def printGridStateValues(V):
    grid = np.zeros([4,4])

    for state, value in V.items():
        x = state[0]
        y = state[1]
        grid[x,y] = value

    print("Value Function--------------------------")
    pprint.pprint(grid)
    print('\n')

# Gets the action that has the highest Q value for this state
def getMaxActionForState(s, Q):
	max_value = float('-inf')
	max_action = None
	for (state, action), value in Q.items():
		if state == s:
			if value > max_value:
				max_action = action
				max_value = value
	return max_action

# Prints the policy as a grid of arrows
def printPolicy(policy):
    grid = np.zeros([4,4])

    for state, action in policy.policy.items():
        x = state[0]
        y = state[1]
        grid[x,y] = action

    # Convert actions to arrows
    arrow_grid = []
    for row_index, row in enumerate(grid):
        arrow_grid_row = []
        for col_index, action in enumerate(row):
            arrow_char = ''
            if (row_index == 0 and col_index == 0) or (row_index == 3 and col_index == 3):
                arrow_grid_row.append(arrow_char)
            else:
                if action == 0:
                    arrow_char = '↑'
                elif action == 1:
                    arrow_char = '→'
                elif action == 2:
                    arrow_char = '↓'
                elif action == 3:
                    arrow_char = '←'
                arrow_grid_row.append(arrow_char)
        arrow_grid.append(arrow_grid_row)

    print("Policy--------------------------")
    pprint.pprint(arrow_grid)
    print('\n')