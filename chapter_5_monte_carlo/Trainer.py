import os
import sys
import matplotlib.pylab as plt
from PIL import Image
from mpl_toolkits.axes_grid1 import ImageGrid
import numpy as np
import math

class Trainer(object):
	"""
	Takes an agent and an env and facilitates running multiple
	runs of training with the agent on the environment. Each run starts with the agent
	from scratch.

	env: training environment

	agent: agent

	epsilon_values: list of epsilon values to be experimented with

	gamma_values: list of gamma values to be experimented with

	num_runs: total number of from-scratch training runs for each combination of parameters

	num_eps_per_run: number of episodes in each training run

	eval_freq: number of episodes between evaluations during training

	"""
	def __init__(self, env, agent, epsilon_values, gamma_values, num_runs, num_eps_per_run, eval_freq):
		super(Trainer, self).__init__()
		self.env = env
		self.agent = agent
		self.epsilon_values = epsilon_values
		assert len(self.epsilon_values) > 0
		self.gamma_values = gamma_values
		assert len(self.gamma_values) > 0
		self.num_runs = num_runs
		self.num_eps_per_run = num_eps_per_run
		self.eval_freq = eval_freq

		print("Initialized Trainer")
		print("epsilon values", epsilon_values)
		print("gamma values", gamma_values)
		print()

	# Train the agent 'num_runs' times, generating log and evaluation output for each run
	def train(self):
		training_data_dir = self.makeTrainingDataDir()
		all_param_dirs = []
		for epsilon in self.epsilon_values:
			for gamma in self.gamma_values:
				param_dir = self.makeParamDir(training_data_dir, epsilon, gamma)
				self.runTrainingForParams(epsilon, gamma, param_dir)
				all_param_dirs.append(param_dir)
		self.genAvgEvalRewardsGrid(training_data_dir, all_param_dirs)

	# Runs num_runs training runs for a set of parameters and log results
	def runTrainingForParams(self, epsilon, gamma, param_dir):
		original_stdout = sys.stdout
		all_eval_rewards = []
		print("Running training for parameters")
		print("epsilon:", epsilon)
		print("gamma:", gamma)
		for run_idx in range(self.num_runs):
			run_dir = param_dir + '/run_' + str(run_idx)
			os.mkdir(run_dir)
			# Create a log file for this training run
			file = open(run_dir + '/logs.txt', 'w')
			sys.stdout = file

			print("Training run", run_idx)
			print("---------------------------")
			self.agent.reset()
			self.agent.epsilon = epsilon
			self.agent.gamma = gamma
			eval_rewards = self.agent.train(self.num_eps_per_run)
			all_eval_rewards.append(eval_rewards)
			self.genEvalRewardPlot(eval_rewards, run_idx, run_dir)

			file.close()

			# Save the policy
			np.save(run_dir + '/policy.npy', self.agent.policy) 

			# Reset the agent so that the next run will be fresh
			self.agent.reset()

		sys.stdout = original_stdout
		self.plotAllEvalRewards(all_eval_rewards, param_dir, epsilon, gamma)
		self.plotAvgEvalRewards(all_eval_rewards, param_dir, epsilon, gamma)
		print()

	# Create the training data directory where all log an eval output will go
	def makeTrainingDataDir(self):
		# Create a new training_data directory each time with an increasing index
		training_data_index = 0
		training_data_dir = ''
		while True:
			training_data_dir = './training_data_' + str(training_data_index)
			if os.path.exists(training_data_dir):
				training_data_index += 1
			else:
				os.mkdir(training_data_dir)
				break
		return training_data_dir

	# Create the directory inside the training_data_dir for the results for
	# this combination of parameters
	def makeParamDir(self, training_data_dir, epsilon, gamma):
		param_dir = training_data_dir + '/e_' + str(epsilon) + '_g_' + str(gamma)
		os.mkdir(param_dir)
		return param_dir

	# Create the plot for evaluation rewards in a single training run
	def genEvalRewardPlot(self, eval_rewards, run_idx, run_dir):
		plt.plot(list(eval_rewards.keys()), list(eval_rewards.values()))
		plt.xlabel("Episode") 
		plt.ylabel("Avg. Cum. Reward")
		plt.title('Eval Rewards for run ' + str(run_idx))
		plt.savefig(run_dir + '/eval_plot.png')  
		plt.clf()

	# Create the plot for evaluation rewards across all training runs
	def plotAllEvalRewards(self, all_eval_rewards, param_dir, epsilon, gamma):
		for run_idx, eval_rewards in enumerate(all_eval_rewards):
			plt.plot(list(eval_rewards.keys()), list(eval_rewards.values()), label='run_'+str(run_idx))
		plt.xlabel("Episode") 
		plt.ylabel("Avg. Cum. Reward")
		plt.title('All Eval Rewards Across Training Runs for e: '
			+ str(epsilon) + ', g: ' + str(gamma))
		leg = plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
		plt.savefig(param_dir + '/all_runs_eval_plot.png', bbox_inches="tight")  
		plt.clf()

	# Create the plot for the avg eval reward at every eval step across all runs
	def plotAvgEvalRewards(self, all_eval_rewards, param_dir, epsilon, gamma):
		avg_eval_rewards_for_each_step = {}

		# Get each of the steps that the agent was evaluated at
		# e.g. if agent is evaluated every 100 steps -> 100, 200, 300, ... 
		eval_steps = list(all_eval_rewards[0].keys())

		# Calculate the average reward the agent had at each eval step across runs
		for eval_step in eval_steps:
			eval_rewards_for_step = [eval_rewards[eval_step] for eval_rewards in all_eval_rewards]
			avg_eval_reward_for_step = sum(eval_rewards_for_step) / len(eval_rewards_for_step)
			avg_eval_rewards_for_each_step[eval_step] = avg_eval_reward_for_step

		plt.ylim(0,1)
		plt.plot(list(avg_eval_rewards_for_each_step.keys()), list(avg_eval_rewards_for_each_step.values()))
		plt.xlabel("Episode") 
		plt.ylabel("Avg. Cum. Reward")
		plt.title('Avg. Eval Reward Across All Training Runs for e: '
			+ str(epsilon) + ', g: ' + str(gamma))
		plt.savefig(param_dir + '/all_runs_avg_eval_plot.png')
		plt.clf()

	# Generates a grid of images from all of the avg eval rewards plots
	# across different parameter combinations
	def genAvgEvalRewardsGrid(self, training_data_dir, all_param_dirs):
		# Get all of the avg eval_reward filenames
		avg_eval_reward_files = list(map(lambda param_dir: param_dir + '/all_runs_avg_eval_plot.png',
			all_param_dirs))

		# Convert the images to np arrays
		np_images = []
		for filename in avg_eval_reward_files:
			img = Image.open(filename)
			np_images.append(np.asarray(img))

		# Plot all of the images into a grid
		grid_size = math.ceil(math.sqrt(len(np_images)))
		fig = plt.figure(figsize=(16., 16.))
		grid = ImageGrid(fig, 111,  # similar to subplot(111)
		                 nrows_ncols=(grid_size, grid_size),  # creates 2x2 grid of axes
		                 axes_pad=0.1,  # pad between axes in inch.
		                 )
		for ax, im in zip(grid, np_images):
		    # Iterating over the grid returns the Axes.
		    ax.imshow(im)
		plt.savefig(training_data_dir + '/avg_eval_rewards_grid.png')  
		plt.clf()


