from Agents import GradientBanditAgent
import matplotlib.pyplot as plt
from NonStationaryBanditEnv import NonStationaryBanditEnv
import numpy as np

env = NonStationaryBanditEnv()

def testGBanditAgent(alpha):
	g_bandit_agent = GradientBanditAgent(env, alpha)
	num_episodes = 10000
	env.reset()
	g_bandit_agent_rewards = []
	for i in range(num_episodes):
	  action = g_bandit_agent.selectAction()
	  obs, reward, done, info = env.step(action)
	  print("Took action:", action, "reward:", reward)
	  g_bandit_agent.updateActionPreferences(action, reward)
	  g_bandit_agent_rewards.append(reward)
	return sum(g_bandit_agent_rewards) / num_episodes


# 0.1, 0.2, ..., 0.9
alphas = np.arange(1, 100, 10)

all_rewards = []
for alpha in alphas:
	rewards = testGBanditAgent(alpha)
	all_rewards.append(rewards)

print(all_rewards)
plt.plot(alphas, all_rewards, marker='o')
plt.xlabel('alpha')
plt.ylabel('avg reward') 
plt.show()