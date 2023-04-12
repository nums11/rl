import gym
from gym import Env
from gym.spaces import Discrete, Tuple
import random
import numpy as np
import pprint

class GridWorldEnv(Env):
    """
	Simple 4x4 GridWorld Env from figure 4.1 in the RL textbook

	Actions are up (0), right (1), down (2), left (3).

	Reward is -1 for every timestep until the terminal states are reached
    at which a reward of 5 is given.

	Actions that take the agent off the grid leave the agent unchanged.
    """
    def __init__(self):
        self.action_space = Discrete(4)
        self.observation_space = Tuple((Discrete(4), Discrete(4)))
        # Initialize the grid
        self.grid = np.zeros([4,4])
        self.grid[0][0] = 1
        self.grid[3][3] = 1
        self.max_steps = 50
        self.step_count = 0

    def step(self, action):
        assert self.action_space.contains(action)

        new_pos = None
        if action == 0: # up
        	new_pos = (self.pos[0] - 1, self.pos[1])
        elif action == 1: # right
        	new_pos = (self.pos[0], self.pos[1] + 1)
        elif action == 2: # down
        	new_pos = (self.pos[0] + 1, self.pos[1])
        elif action == 3: # left
        	new_pos = (self.pos[0], self.pos[1] - 1)

        # Make sure the agent doesn't fall off of the grrid
        if self._onGrid(new_pos):
        	self.pos = new_pos

        # Check if the agent is in a terminal position
        # Negative reward for every timestep except for terminal step
        done = False
        reward = -1
        if self._inTerminalPos():
            done = True
            reward = 5

        if self.step_count == self.max_steps:
            done = True

        self.step_count += 1

        return self.pos, reward, done, {}

    def _onGrid(self, pos):
    	return pos[0] in range(0,4) and pos[1] in range(0,4)

    def _inTerminalPos(self):
    	return self.grid[self.pos[0]][self.pos[1]] == 1

    def reset(self):
        # Reset step count
        self.step_count = 0
        # Start the agent in the top right corner
        self.pos = (0,3)
        return self.pos

    # Optionally allow start position to be set
    def setStartPos(self, pos):
        if self._onGrid(pos):
            self.pos = pos
        else:
            print("ERROR: Start position not on grid")

    def render(self):
        # Convert every element in the grid from a number to a string
        new_grid = list(
            map(lambda r:
                list(map(lambda c: str(c), r)),
            self.grid)
        )

        # Put an X where the agent currrently is
        new_grid[self.pos[0]][self.pos[1]] = 'X'

        pprint.pprint(new_grid)

