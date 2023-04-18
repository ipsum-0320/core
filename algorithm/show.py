import sys
sys.dont_write_bytecode = True

import matplotlib.pyplot as plt
import numpy as np
import os

cwd_path = os.getcwd()

if __name__ == "__main__":
  # show.py 用于展示不同算法的效果。

  BBO_Pro = np.loadtxt(os.path.join(cwd_path, "./algorithm/data/BBO-Pro.txt")).T.tolist()
  BBO = np.loadtxt(os.path.join(cwd_path, "./algorithm/data/BBO.txt")).T.tolist()
  GA = np.loadtxt(os.path.join(cwd_path, "./algorithm/data/GA.txt")).T.tolist()


  fig, ax = plt.subplots()
  
  ax.plot(BBO_Pro[0], BBO_Pro[1], label="BBO-Pro_min")
  ax.plot(BBO[0], BBO[1], label="BBO_min")
  ax.plot(GA[0], GA[1], label="GA_min")

  ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True)) # 显示整数刻度。

  ax.set_title("Performance of Minimum \nAckley Function Value")
  ax.set_xlabel("Iterations")
  ax.set_ylabel("Ackley Function Minimum Value")
  ax.legend()

  plt.show()
