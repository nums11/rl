from Agents import UCBAgent
import matplotlib.pyplot as plt
from NonStationaryBanditEnv import NonStationaryBanditEnv
import numpy as np

env = NonStationaryBanditEnv()

def testUCBAgent(c):
	ucb_agent = UCBAgent(env, 0.1, c)
	num_episodes = 10000
	env.reset()
	ucb_agent_rewards = []
	for i in range(num_episodes):
	  action = ucb_agent.selectAction(i)
	  obs, reward, done, info = env.step(action)
	  print("Took action:", action, "reward:", reward)
	  ucb_agent.updateActionEstimation(action, reward)
	  ucb_agent_rewards.append(reward)
	return sum(ucb_agent_rewards) / num_episodes


# 0.1, 0.2, ..., 0.9
confidence_values = np.arange(0.1, 1, 0.1)

all_rewards = []
for c in confidence_values:
	rewards = testUCBAgent(c)
	all_rewards.append(rewards)

print(all_rewards)
plt.plot(confidence_values, all_rewards, marker='o')
plt.xlabel('c')
plt.ylabel('avg reward') 
plt.show()