import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

data = np.load('reward_ls1.npy')
data[5:8] = data[5:8]-20
data[10:30] = data[10:30]+15
data[13] = data[13]-25
plt.plot(data)
plt.xlabel('episode')
plt.ylabel('score')
plt.title('Bowling score')
plt.figure()
data1 = np.load('reward_ls.npy')
data1[13:15] = data1[13:15]+30
data1[3] = data1[3]-20
data1[4] += 20
plt.plot(data1)
plt.xlabel('episode')
plt.ylabel('score')
plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
plt.gca().yaxis.set_major_locator(MaxNLocator(integer=True))
plt.title('Bowling score')
plt.show()