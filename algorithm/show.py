import sys
sys.dont_write_bytecode = True

import matplotlib.pyplot as plt
import numpy as np
import os

cwd_path = os.getcwd()

if __name__ == "__main__":
  # show.py 用于展示不同算法的效果。

  BBO_Pro_min = np.loadtxt(os.path.join(cwd_path, "./algorithm/data/BBO-Pro_min.txt")).T.tolist()
  BBO_min = np.loadtxt(os.path.join(cwd_path, "./algorithm/data/BBO_min.txt")).T.tolist()
  BBO_Pro_avg = np.loadtxt(os.path.join(cwd_path, "./algorithm/data/BBO-Pro_avg.txt")).T.tolist()
  BBO_avg = np.loadtxt(os.path.join(cwd_path, "./algorithm/data/BBO_avg.txt")).T.tolist()
  GA_min = np.loadtxt(os.path.join(cwd_path, "./algorithm/data/GA_min.txt")).T.tolist()
  GA_avg = np.loadtxt(os.path.join(cwd_path, "./algorithm/data/GA_avg.txt")).T.tolist()

  fig, (ax_min, ax_avg) = plt.subplots(1, 2)
  
  ax_min.plot(BBO_Pro_min[0], BBO_Pro_min[1], label="BBO-Pro")
  ax_min.plot(BBO_min[0], BBO_min[1], label="BBO")
  ax_min.plot(GA_min[0], GA_min[1], label="GA")

  ax_avg.plot(BBO_Pro_avg[0], BBO_Pro_avg[1], label="BBO-Pro")
  ax_avg.plot(BBO_avg[0], BBO_avg[1], label="BBO")
  ax_avg.plot(GA_avg[0], GA_avg[1], label="GA")


  ax_min.xaxis.set_major_locator(plt.MaxNLocator(integer=True)) # 显示整数刻度。
  ax_avg.xaxis.set_major_locator(plt.MaxNLocator(integer=True))

  ax_min.set_title("Performance of Minimum \nAckley Function Value")
  ax_min.set_xlabel("Iterations")
  ax_min.set_ylabel("Ackley Function Minimum Value")
  ax_min.legend()

  ax_avg.set_title("Performance of Average \nAckley Function Value")
  ax_avg.set_xlabel("Iterations")
  ax_avg.set_ylabel("Ackley Function Average Ackley Value")
  ax_avg.legend()

  plt.show()
