import gym
import gym_bandits
from Agents import GreedyAgent, EpsilonGreedyAgent
import matplotlib.pyplot as plt

# env = gym.make("BanditTwoArmedDeterministicFixed-v0")
env = gym.make("BanditTenArmedUniformDistributedReward-v0")

greedy_agent = GreedyAgent(env)
e_greedy_agent = EpsilonGreedyAgent(env, 0.1)

env.reset()
num_episodes = 1000
greedy_agent_rewards = []
for i in range(num_episodes):
  action = greedy_agent.selectAction()
  obs, reward, done, info = env.step(action)
  print("Took action:", action, "reward:", reward)
  greedy_agent.updateActionEstimation(action, reward)
  greedy_agent_rewards.append(reward)

e_greedy_agent_rewards = []
for i in range(num_episodes):
  action = e_greedy_agent.selectAction()
  obs, reward, done, info = env.step(action)
  print("Took action:", action, "reward:", reward)
  e_greedy_agent.updateActionEstimation(action, reward)
  e_greedy_agent_rewards.append(reward)

print("Results after", num_episodes, "episodes")
print("-----------------------------------")
print("Cumulative reward:")
print("Greedy:", sum(greedy_agent_rewards), "e_greedy", sum(e_greedy_agent_rewards))
print("Average rewards:")
print("Greedy:", sum(greedy_agent_rewards) / len(greedy_agent_rewards),
	"e_greedy", sum(e_greedy_agent_rewards) / len(e_greedy_agent_rewards))

figure, axis = plt.subplots(1, 2)
axis[0].plot(greedy_agent_rewards)
axis[0].set_title("Greedy")
axis[1].plot(e_greedy_agent_rewards)
axis[1].set_title("E-Greedy")

plt.show()
