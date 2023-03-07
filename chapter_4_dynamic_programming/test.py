import gym
from GridWorldEnv import GridWorldEnv
from Agents import PolicyIterationAgent, ValueIterationAgent

env = GridWorldEnv()
env.reset()

pi_agent = PolicyIterationAgent(0.9)
pi_agent.policyIterate()

vi_agent = ValueIterationAgent(0.9)
vi_agent.valueIterate()
