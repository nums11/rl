from GridWorldEnv import GridWorldEnv
from Helpers import *
from tqdm import tqdm

env = GridWorldEnv()

num_episodes = 10000
gamma = 0.9

# Implementation of first visit MC prediction
def firstVisitMCPred(gamma, num_episodes):
	# Initialize nondet policy that selects an action
	# randomly with all actions having equal probability
	policy = Policy('nd')

	# Initialize V (state-value function) and returns for the 4x4 grid world
	V = {}
	returns = {}
	for x in range(4):
		for y in range(4):
			V[(x,y)] = 0 # Set initial state values to 0
			returns[(x,y)] = []

	# loop number of episodes
	for _ in tqdm(range(num_episodes)):
		# Generate an episode and receive the trajectory
		trajectory = genEpisode(env, policy)
		# Initialize G - discounted return
		G = 0
		# Iterate through trajectory in reverse
		for t in reversed(trajectory):
			# Grab state, action, and reward from this step of the trajectory
			s_t, a_t, r_t_plus_1 = trajectory[t]
			# Add to the discounted return
			G = gamma * G + r_t_plus_1
			# Store return and update state value function if this is the first
			# visit of this state
			if isFirstVist(s_t, t, trajectory):
				returns[s_t].append(G)
				V[s_t] = average(returns[s_t])

	printGridStateValues(V)

# Uncomment below to run the algo
# firstVisitMCPred(gamma, num_episodes)

# Implementation of Monte Carlo Control with Exploring Starts
def MCES(gamma, num_episodes):
	# Initialize deterministic policy with initial actions chosen
	# randomly for each state
	policy = Policy('d')

	# Initialize Q (action-value function) and returns for the 4x4 grid world
	Q = {}
	returns = {}
	for x in range(4):
		for y in range(4):
			for action in range(4):
				Q[((x,y), action)] = 0 # Initial action values are 0
				returns[((x,y), action)] = []

	# loop number of episodes
	for _ in tqdm(range(num_episodes)):
		# Generate an episode with explorint starts and receive the trajectory
		trajectory = genEpisode(env, policy, es=True)
		# Initialize G - discounted return
		G = 0
		# Iterate through trajectory in reverse
		for t in reversed(trajectory):
			# Grab state, action, and reward from this step of the trajectory
			s_t, a_t, r_t_plus_1 = trajectory[t]
			# Add to the discounted return
			G = gamma * G + r_t_plus_1
			# Store return and update action value function and policy if this is the first
			# visit of this state
			if isFirstVist(s_t, t, trajectory):
				returns[(s_t, a_t)].append(G)
				Q[(s_t, a_t)] = average(returns[(s_t, a_t)])
				policy.updateDetPolicyState(s_t, getMaxActionForState(s_t, Q))

	printPolicy(policy)

# Uncomment below to run the algo
# MCES(gamma, num_episodes)

def OnPolicyFirstVistMCControl(gamma, num_episodes, epsilon=0.1):
	# Initialize e-greedy policy with initial actions chosen
	# randomly for each state
	policy = Policy('eg', epsilon)

	# Initialize Q (action-value function) and returns for the 4x4 grid world
	Q = {}
	returns = {}
	for x in range(4):
		for y in range(4):
			for action in range(4):
				Q[((x,y), action)] = 0 # Initial action values are 0
				returns[((x,y), action)] = []

	# loop number of episodes
	for _ in tqdm(range(num_episodes)):
		# Generate an episode and receive the trajectory
		trajectory = genEpisode(env, policy)
		# Initialize G - discounted return
		G = 0
		# Iterate through trajectory in reverse
		for t in reversed(trajectory):
			# Grab state, action, and reward from this step of the trajectory
			s_t, a_t, r_t_plus_1 = trajectory[t]
			# Add to the discounted return
			G = gamma * G + r_t_plus_1
			# Store return and update action value function and policy if this is the first
			# visit of this state
			if isFirstVist(s_t, t, trajectory):
				returns[(s_t, a_t)].append(G)
				Q[(s_t, a_t)] = average(returns[(s_t, a_t)])
				policy.updateDetPolicyState(s_t, getMaxActionForState(s_t, Q))

	printPolicy(policy)

# Uncomment below to run the algo
# OnPolicyFirstVistMCControl(gamma, num_episodes)

# An incremental updates version of off-policy MC Control
# using weighted importance sampling
def OffPolicyEveryVistMCControl(gamma, num_episodes):
	# Target policy is a deterministic policy with initial actions chosen
	# randomly for each state
	pi = Policy('d')
	# Behavior policy is a nondeterministic policy that selects all actions
	# with equal probability
	b = Policy('nd')

	# Initialize Q: action value function
	# and C: cumulative sum of weights for every state-action pair
	# for the 4x4 grid world
	Q = {}
	C = {}
	for x in range(4):
		for y in range(4):
			for action in range(4):
				Q[((x,y), action)] = 0 # Initial action values are 0
				C[((x,y), action)] = 0 # Initial sums are set to 0

	# loop number of episodes
	for _ in tqdm(range(num_episodes)):
		# Generate an episode following the behavior policy and receive the trajectory
		trajectory = genEpisode(env, b)
		# Initialize G - discounted return
		G = 0
		# Initialize W - importance sampling weight (ratio)
		W = 1
		# Iterate through trajectory in reverse
		for t in reversed(trajectory):
			# Retrieve state, action, and reward from this step of the trajectory
			s_t, a_t, r_t_plus_1 = trajectory[t]
			# Add to the discounted return
			G = gamma * G + r_t_plus_1
			# Add to the sum of weights for this state action pair
			C[((x,y), action)] += W
			# Update the action-values for our target policy
			Q[(s_t, a_t)] += (W / C[((x,y), action)]) * (G - Q[(s_t, a_t)])
			# Make the target policy greedy with respect to Q
			pi.updateDetPolicyState(s_t, getMaxActionForState(s_t, Q))
			# If action that was taken in this trajectory is not equal to
			# the action that the target policy would have taken, skip the rest
			# of this episode. Since the target policy is deterministic the likelihood
			# of the rest of the trajectory occuring would be 0.
			if a_t != pi.policy[s_t]:
				break
			# Weigh the likelihood of this trajectory from the behavior policy occuring
			# in the target. Since the target is deterministic, pi(a_t|s_t) = 1
			W *= 1 / b.policy[s_t][a_t]

	printPolicy(pi)

# Uncomment below to run the algo
# OffPolicyEveryVistMCControl(gamma, num_episodes)