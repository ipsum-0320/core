import sys

sys.dont_write_bytecode = True

import os
import json
import random
import math
from typing import TypedDict, List
from decimal import Decimal, getcontext
from tqdm import tqdm
import numpy as np



getcontext().prec = 112   # 设置精度为 112

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


def roulette(origin_list): # 轮盘赌算法。
  origin_sum = sum(origin_list) 
  p_list = [ item / origin_sum for item in origin_list ]
  p_accumulate_list = [ sum(p_list[0:(i + 1)]) for i in range(0, len(p_list)) ]
  target = random.random()
  (start, end) = (0, len(p_accumulate_list) - 1)
  while start + 1 < end:
    mid = start + ((end - start) >> 1);
    if target > p_accumulate_list[mid]: start = mid
    else: end = mid
  if target <= p_accumulate_list[start]: return start
  else: return end

class BBO:
  def __init__(self, json):
    self.info: Info = {
      "nodes": None,
      "tasks": [],
    }
    self.info["nodes"] = json["nodes"]
    for val in json["tasks"]:
      for i in range(1, int(val["nums"]) + 1):
        task: Task = {
          "podName": "%s-%d" % (val["podName"][:-4], i),
          "image": val["image"],
          "calcMetrics": int(val["calcMetrics"])
        }
        self.info["tasks"].append(task)

    # 定义参数。
    self.iterations = 80 # 算法的迭代次数。
    self.solution_size = 50 # 种群中栖息地的数量，这里指代解的数量。
    self.S_max = 50 # 种群数量的最大值，用于计算迁入迁出率，为了能够同时去到 0 和 1，取 S_max = solution_size - 1。
    self.vector_size = len(self.info["tasks"]) # 解向量的长度。
    self.move_in_max = 1 # 迁入率最大值。
    self.move_out_max = 1 # 迁出率最大值。
    self.mutation_p = 0.01 # 变异率。
    self.node_quantity = 6 # 可用于运行 pod 的工作节点数量。
    self.time_weight = [round(0.85 + x / 100, 2) for x in range(0, 6)] # 为了能够尽可能取得最优解，选择多个权向量确定不同的搜索方向。
    self.logistics_K = 0.000025 # logistics 函数中的 K 值，参数的确定基于函数图像的调整。
    self.logistics_X_0 = 300000 # logistics 函数中的 X_0 值，参数的确定基于函数图像的调整。
    self.task_calc_density = 23 # 任务的计算密度。
    self.trans_bandwidth = 20.29 # 传输带宽。
    self.energy_density = 11.3453 # 任务的能量密度。
    self.e_per_time = self.task_calc_density * self.trans_bandwidth # 单位时间的传输所消耗的能量。
    self.mutation_to_node00_list_len = 8

    # 定义变量。
    self.solutions = [] # 解向量。
    self.solution_id_map = {} # 用于完成 id 和 solution 的映射。
    self.task_trans_metrics = [task["calcMetrics"] for task in self.info["tasks"]] # 任务的输入量。
    self.task_calc_metrics = [self.task_calc_density * task["calcMetrics"] for task in self.info["tasks"]] # 任务的计算量。
    self.calc_abilities = self.get_calc_ability() # node 的计算能力。
    self.data_min = [] # 存储迭代过程中的最优值。

    # 用于归一化的参数。
    self.min_T = self.get_min_T() # T 的最小值，用于归一化。
    self.max_T = self.get_max_T() # T 的最大值，用于归一化。
    self.min_E = sum([self.energy_density * calc for calc in self.task_calc_metrics]) # E 的最小值，用于归一化。
    self.max_E = self.min_E + sum([self.e_per_time * tran / self.trans_bandwidth for tran in self.task_trans_metrics]) # E 的最大值，用于归一化。

    # 初始化。
    for i in range(0, self.solution_size):
      solution = {
        "id": i, # 给解进行编号。
        "HSI": None,
        "vector": [random.randint(0, 5) for _ in range(0, self.vector_size)], # 设置定义域。
        "move_in": None,
        "move_out": None,
      }
      self.solutions.append(solution)
      self.solution_id_map[i] = solution
      self.get_HSI(solution)

    # 迭代。
    self.solutions.sort(key=lambda el: el["HSI"])
    for i in tqdm(range(0, self.iterations)):
      # 为了方便迁移率的计算，按照 HSI 的值升序排序，越靠前，解越好。
      for j in range(0, self.solution_size):
        # 计算迁移率。
        self.get_move(self.solutions[j], j)
      for solution in self.solutions:
        self.move(solution)
        self.mutation(solution)
      self.solutions.sort(key=lambda el: el["HSI"])
      self.data_min.append([i + 1, self.solutions[0]["HSI"]])
    
    cwd_path = os.getcwd()
    np.savetxt(os.path.join(cwd_path, "./application/data/BBO.txt"), np.array(self.data_min), header="Iteration Cost",  fmt="%d %f")

  def get_min_T(self):
    # 计算 min_T，采取充分利用假说来估计 min_T。
    sum_ability = sum(self.calc_abilities) # 总计算能力。
    sum_calc = sum(self.task_calc_metrics) # 总计算量。
    t_1 = sum_calc / sum_ability # 理想的计算时延。
    node00_alpha = self.calc_abilities[0] / sum_ability # node00 计算能力占比，可以等价为其处理的计算量占比。
    sum_trans = node00_alpha * sum(self.task_trans_metrics) # 传到到中心云的数据量。 
    t_2 = sum_trans / self.trans_bandwidth # 理想的传输时延。
    return t_1 + t_2

  def get_max_T(self):
    # 计算 max_T，计算方法是将所有的任务分配给计算能力最差的节点。
    min_calc_ability = min(self.calc_abilities)
    min_index = self.calc_abilities.index(min_calc_ability)
    vector = [min_index for _ in range(0, self.vector_size)]
    tasks_assigned_per_node = [0, 0, 0, 0, 0, 0] 
    for el in vector:
      tasks_assigned_per_node[el] = tasks_assigned_per_node[el] + 1
    T = 0
    for i in range(0, self.vector_size):
      T = T + self.task_calc_metrics[i] / (self.calc_abilities[vector[i]] / tasks_assigned_per_node[vector[i]])
    return T
    
  def get_calc_ability(self):
    # 计算 node 的计算能力。
    calc_abilities = []
    for node in self.info["nodes"]:
      calc_abilities.append(round((1 / (1 + (math.e ** -(self.logistics_K * (node["mem"] - self.logistics_X_0))))) * node["cpu"], 2))
      # 使用 logistics 函数计算。
    return calc_abilities

  def get_HSI(self, solution):
    # 计算 HSI，HSI 值越小，那么解越优质。
    HSI = sys.maxsize
    T = 0
    E = 0
    tasks_assigned_per_node = [0, 0, 0, 0, 0, 0] 
    # 存储每个节点分配的任务个数。
    for el in solution["vector"]:
      tasks_assigned_per_node[el] = tasks_assigned_per_node[el] + 1
    for i in range(0, len(solution["vector"])):
      vector_el = solution["vector"][i]
      T = T + self.task_calc_metrics[i] / (self.calc_abilities[vector_el] / tasks_assigned_per_node[vector_el]) # 给节点重复分配任务会导致运行时间提高。
      E = E + self.energy_density * self.task_calc_metrics[i] # 给节点重复分配任务不会导致运行能耗改变。
      if vector_el == 0:
        T = T + self.task_trans_metrics[i] / self.trans_bandwidth # 给节点重复分配任务不会导致传输时间改变。
        E = E + self.e_per_time * self.task_trans_metrics[i] / self.trans_bandwidth # 给节点重复分配任务不会导致传输能耗改变。
    normalized_T = Decimal((T - self.min_T) / (self.max_T - self.min_T)) # 归一化的 T。
    normalized_E = Decimal((E - self.min_E) / (self.max_E - self.min_E)) # 归一化的 E。
    for weight in self.time_weight:
      HSI = min(HSI, Decimal(weight) * normalized_T + Decimal(1 - weight) * normalized_E)
    solution["HSI"] = HSI

  def get_move(self, solution, index):
    # 计算迁入迁出率。
    s = self.S_max - index # 实现 s 和解的映射。
    solution["move_in"] = (self.move_in_max / 2) * (math.cos((s * math.pi) / self.S_max) + 1)
    solution["move_out"] = (self.move_out_max / 2) * (-math.cos((s * math.pi) / self.S_max) + 1)

  def get_solution(self):
    best_solution = self.solutions[0]["vector"]
    print("best_cost: ", self.solutions[0]["HSI"])
    print()
    print("best_solution: ")
    best_solution_map_list = []
    for i in range(0, len(self.info["tasks"])):
      best_solution_map = {
        "podName": self.info["tasks"][i]["podName"],
        "image": self.info["tasks"][i]["image"],
        "nodeName": "node0%d" % best_solution[i]
      }
      best_solution_map_list.append(best_solution_map)
    return best_solution_map_list

  def move(self, solution):
    # 迁移。
    copy_solution = solution.copy() # 执行精英保存策略。
    for i in range(0, self.vector_size):
      if random.random() < solution["move_in"]:
        other_move_out_list = []
        other_id_list = []
        for s in self.solutions:
          if s["id"] != solution["id"]:
            other_move_out_list.append(s["move_out"])
            other_id_list.append(s["id"])
        target_solution = self.solution_id_map[other_id_list[roulette(other_move_out_list)]]
        solution["vector"][i] = target_solution["vector"][i]
    self.get_HSI(solution)
    if solution["HSI"] > copy_solution["HSI"]: # 得到劣化解。
      solution["HSI"] = copy_solution["HSI"]
      solution["vector"] = copy_solution["vector"]

  def mutation(self, solution):
    # 变异。
    copy_solution = solution.copy() # 精英保存策略。
    for i in range(0, self.vector_size):
      if random.random() < self.mutation_p:
        node = random.randint(-(self.mutation_to_node00_list_len - 1), self.node_quantity - 1) 
        solution["vector"][i] = 0 if node <= 0 else node
    self.get_HSI(solution)
    if solution["HSI"] > copy_solution["HSI"]: # 得到劣化解。
      solution["HSI"] = copy_solution["HSI"]
      solution["vector"] = copy_solution["vector"]
    

if __name__ == "__main__":
  cwd_path = os.getcwd()
  with open(os.path.join(cwd_path, "./application/mock.json"), "r") as mock:
    mock_json = json.load(mock)
  print() # 空行。
  solution = BBO(mock_json)
  print() # 空行。
  for s in solution.get_solution():
    print(s)