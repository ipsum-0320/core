import json
import random
import os
import math
import sys
sys.dont_write_bytecode = True

from typing import TypedDict, List
import numpy as np

class Node:
  nodeName: str
  cpu: float
  mem: float

class Task:
  podName: str
  image: str
  calcMetrics: str

class Info(TypedDict):
  nodes: List[Node]
  tasks: List[Task]

def normalized(x):
  # 使用反正切函数 atan 做数值归一化。
  return math.atan(x) * (2 / math.pi)

class BBO:
  def __init__(self, json): # 所有属性的定义都应该在 __init__ 中。
    # 存储初始信息。
    self.info: Info = {
      "nodes": None,
      "tasks": [],
    }
    self.info["nodes"] = json["nodes"]
    for val in json["tasks"]:
      for i in range(1, int(val["nums"]) + 1):
        task: Task = {
          "podName": "%s-%d" % (val["podName"], i),
          "image": val["image"],
          "calcMetrics": int(val["calcMetrics"])
        }
        self.info["tasks"].append(task)
    
    # 定义参数。
    self.iterations = 500 # 算法的迭代次数。
    self.population_size = 8 # 种群中栖息地的数量，这里指代解的数量。
    self.S_max = 7 # 种群数量的最大值，用于计算迁入迁出率，为了能够同时去到 0 和 1，取 S_max = population_size - 1。
    self.vector_size = len(self.info["tasks"]) # 解向量的长度。
    self.node_quantity = 6 # 可用于运行 pod 的工作节点数量。
    self.calc_metrics_threshold = 300 # 初始化时，计算任务被分配到 node00 上的计算量阈值。
    self.time_weight = [round(0.6 + x / 100, 2) for x in range(0, 11)] # 为了能够尽可能取得最优解，选择多个权向量确定不同的搜索方向，其中 time 的权重范围为 [0.6, 0.7]。
    self.logistics_K = 0.000025 # logistics 函数中的 K 值，参数的确定基于函数图像的调整。
    self.logistics_X_0 = 300000 # logistics 函数中的 X_0 值，参数的确定基于函数图像的调整。
    self.task_calc_density = 23 # 任务的计算密度。
    self.trans_bandwidth = 20.29 # 传输带宽。
    self.e_per_time = 3205250 # 单位时间的传输所消耗的能量。
    self.move_in_max = 1 # 迁入率的最大值。
    self.move_out_max = 1 # 迁出率的最大值。

    # 定义变量。
    self.solutions = [] # 一个复杂的数据结构，包含解、迁入迁出率和 HSI 值。
    self.task_trans_metrics = [task["calcMetrics"] for task in self.info["tasks"]] # 任务的输入量。
    self.task_calc_metrics = [self.task_calc_density * task["calcMetrics"] for task in self.info["tasks"]] # 任务的计算量。
    self.calc_abilities = self.get_calc_ability() # node 的计算能力。

    for i in range(0, self.population_size):
      self.solutions.append({
        "HSI": 0.0,
        "vector": np.zeros(self.vector_size, dtype = int),
        "move_in": 0.0,
        "move_out": 0.0,
      })
    
    # 初始化种群。
    for i in range(0, self.population_size):
      for j in range(0, self.vector_size):
        self.solutions[i]["vector"][j] = random.randint(0, self.node_quantity - 1)
        if self.info["tasks"][j]["calcMetrics"] > self.calc_metrics_threshold:
          self.solutions[i]["vector"][j] = 0 # 将计算量较大的任务初始分配给 node00。  
      self.get_HSI(self.solutions[i])

    self.solutions.sort(key=lambda el: el["HSI"], reverse=True) # 按照 HSI 的值降序排序，越靠前，解越差。
    for i in range(0, self.population_size):
      self.get_move(self.solutions[i], i)

    # 开始迭代算法。
    for i in range(0, self.iterations):
      i

    

  def get_move(self, solution, index):
    # 计算迁入迁出率。
    s = self.S_max - index # 实现 s 和解的映射。
    solution["move_in"] = (self.move_in_max / 2) * (math.cos((s * math.pi) / self.S_max) + 1)
    solution["move_out"] = (self.move_out_max / 2) * (-math.cos((s * math.pi) / self.S_max) + 1)

  def get_HSI(self, solution):
    # 计算 HSI，HSI 值越小，那么解越优质。
    HSI = sys.maxsize
    T = 0
    E = 0
    for i in range(0, len(solution["vector"])):
      vector_el = solution["vector"][i]
      T = T + self.task_calc_metrics[i] / self.calc_abilities[vector_el]
      E = E + self.task_calc_metrics[i] * self.calc_abilities[vector_el] ** 2
      if vector_el == 0:
        T = T + self.task_trans_metrics[i] / self.trans_bandwidth
        E = E + self.e_per_time * self.task_trans_metrics[i] / self.trans_bandwidth
    for weight in self.time_weight:
      HSI = min(HSI, weight * normalized(T) + (1 - weight) * normalized(E))
    solution["HSI"] = HSI
  
  def get_calc_ability(self):
    # 计算 node 的计算能力。
    calc_abilities = []
    for node in self.info["nodes"]:
      calc_abilities.append(round((1 / (1 + (math.e ** -(self.logistics_K * (node["mem"] - self.logistics_X_0))))) * node["cpu"], 2))
      # 使用 logistics 函数计算。
    return calc_abilities
  
  

if __name__ == "__main__":
  # 读取数据
  cwd_path = os.getcwd()
  with open(os.path.join(cwd_path, "./BBO/mock.json"), "r") as mock:
    mock_json = json.load(mock)
  
  BBO(mock_json)
  



