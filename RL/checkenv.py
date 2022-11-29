from RL.envs.master_state_master_reward import MasterStateMasterReward

env = MasterStateMasterReward()

# check_env(env)

episodes = 1

for episode in range(episodes):
	done = False
	obs = env.reset()
	while not env.data.finished:#not done:
		random_action = env.action_space.sample()
		print("action",random_action)
		state, reward, done, info = env.step(random_action)
		print('reward',reward)

print('checking done!')