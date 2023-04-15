import sys
sys.dont_write_bytecode = True

import math

# 该函数首次出现在论文《A Connectionist Machine for Genetic Hillclimbing》中。

ackley_max = 20
ackley_min = 0

def ackley_generator(vector_size):
  def ackley(X):
    # 基准测试函数 Ackley。
    # 关于这个函数，有如下特性：
    # 1.定义域，该函数的 X_i 通常在 [-32768, 32768] 上进行计算，但是为了提高收敛速度，我们将定义域限制在 [-1000. 1000]。
    # 2.该函数的全局最小值为 0，在 X = (0, 0, ..., 0) 处取到。另外，在有限的定义域内，例如在 [-32768, 32768] 内，可以取到 20 的上界。
    a = 20
    b = 0.2
    c = 2 * math.pi
    sum1 = 0
    sum2 = 0
    for i in range(0, vector_size):
        x = X[i]
        sum1 = sum1 + x ** 2
        sum2 = sum2 + math.cos(c * x)
    term1 = -a * math.exp(-b * math.sqrt(sum1 / 2))
    term2 = -math.exp(sum2 / 2)
    y = term1 + term2 + a + math.exp(1)
    return y 
  return ackley
