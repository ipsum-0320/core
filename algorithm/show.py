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

  fig, ax = plt.subplots()
  
  ax.plot(BBO_Pro[0], BBO_Pro[1], label="BBO-Pro")
  ax.plot(BBO[0], BBO[1], label="BBO")

  ax.set_title("Performance of the algorithm on the Ackley function")
  ax.set_xlabel("iterations")
  ax.set_ylabel("Ackley function values")

  ax.legend()
  plt.show()
