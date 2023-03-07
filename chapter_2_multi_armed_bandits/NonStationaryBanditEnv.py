import gym
from gym import Env
from gym.spaces import Discrete
import random

class NonStationaryBanditEnv(Env):
    """
    Non-Stationary 5-arm Bandit Environment

    At any given time, the best action returns a reward of 1
    while all other actions return a reward of 0.

    Every 500 steps the best action is randomly switched.
    """
    def __init__(self):
        self.action_space = Discrete(5)
        self.observation_space = Discrete(1)

    def step(self, action):
        assert self.action_space.contains(action)

        reward = 0
        done = True

        if action == self.best_action:
            reward = 1

        self.step_count += 1
        # Change the best action every 500 steps
        if self.step_count % 500 == 0:
            self._selectNewBestAction()

        return 0, reward, done, {}

    def _selectNewBestAction(self):
        self.best_action = random.randint(0,4)

    def reset(self):
        self.step_count = 0
        self._selectNewBestAction()

    def render(self, mode='human', close=False):
        pass