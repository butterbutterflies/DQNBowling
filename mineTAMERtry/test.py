import gym


env = gym.make("Bowling-v0").unwrapped
print(env.action_space)
