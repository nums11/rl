import gym
import gym_bandits
from Agents import GreedyAgent, EpsilonGreedyAgent

# env = gym.make("BanditTwoArmedDeterministicFixed-v0")
env = gym.make("BanditTenArmedUniformDistributedReward-v0")

# agent = GreedyAgent(env)
agent = EpsilonGreedyAgent(env, 0.1)

env.reset()
num_episodes = 1000
rewards = []
for i in range(num_episodes):
  action = agent.selectAction()
  obs, reward, done, info = env.step(action)
  print("Took action:", action, "reward:", reward)
  agent.updateActionEstimation(action, reward)
  rewards.append(reward)

print("Results after", num_episodes, "episodes")
print("-----------------------------------")
print("Cumulative reward:", sum(rewards))
print("Average reward:", sum(rewards) / len(rewards))
