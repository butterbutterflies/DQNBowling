import numpy as np
import matplotlib.pyplot as plt


x = np.linspace(1, 30, num=29)
reward = np.load("reward_ls_union.npy")
reward2 = np.load("reward_ls.npy")
plt.plot(x, reward[:-1], label="DQN+Feedback")
plt.plot(x, reward2[:-1], label="DQN")
plt.xlabel("Episode")
plt.ylabel("Reward")
plt.grid()
plt.legend()
plt.title("Reward during training")
plt.savefig("dqnTamer_dqn.png")
plt.show()
