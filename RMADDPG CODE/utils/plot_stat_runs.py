import pickle as pkl
import numpy as np
import matplotlib.pyplot as plt

stat_runs = 15
rrange = [1, 1001, 2001, 3001, 4001, 5001, 6001, 7001,
          8001, 9001]

# file1
filename = "/home/parallels/python_ws/maddpg-pytorch-master/models/simple_spread/Exp/run68/stat_runs"

# file2
filename2 = "/home/parallels/python_ws/AutonomousAgents/ProjectFromScratch/models/simple_spread/Exp/run63/stat_runs"

# unpickle and math
pickle_off = open(filename, "rb")
stat_all = pkl.load(pickle_off)
mean = np.mean(stat_all, axis=1)
std = np.std(stat_all, axis=1) / np.sqrt(stat_runs)
plt.errorbar(rrange, mean, yerr=std)

pickle_off = open(filename2, "rb")
stat_all = pkl.load(pickle_off)
mean = np.mean(stat_all, axis=1)
std = np.std(stat_all, axis=1) / np.sqrt(stat_runs)
plt.errorbar(rrange, mean, yerr=std)

# Plot
# plt.xlim(-1000, 10000)
# plt.ylim(-6, 0)
plt.title("No RNN collide_difference, vs RNN collide_diff. 10 stat runs")
plt.show()
