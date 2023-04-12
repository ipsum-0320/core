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

def roulette(origin_list):
  origin_sum = sum(origin_list) 
  p_list = [ item / origin_sum for item in origin_list ]
  p_accumulate_list = [ sum(p_list[0:(i + 1)]) for i in range(0, len(p_list)) ]
  target = random.random()
  # 使用二分法寻找第一个比 target 大的 p_accumulate_list[i]，效率更高。
  (start, end) = (0, len(p_accumulate_list) - 1)
  while start + 1 < end:
    mid = start + ((end - start) >> 1);
    if target > p_accumulate_list[mid]: start = mid
    else: end = mid
  if target <= p_accumulate_list[start]: return start
  else: return end

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
    self.iterations = 1000 # 算法的迭代次数。
    self.population_size = 50 # 种群中栖息地的数量，这里指代解的数量。
    self.S_max = 49 # 种群数量的最大值，用于计算迁入迁出率，为了能够同时去到 0 和 1，取 S_max = population_size - 1。
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
    self.neighbor_num = 5 # 每个栖息地的相邻栖息地个数。
    self.maturity_max = 0.75 # 成熟度的最大值。
    self.maturity_min = 0.3 # 成熟度的最小值。

    # 定义变量。
    self.solutions = [] # 一个复杂的数据结构，包含 id、解、迁入迁出率和 HSI 值。
    self.solutions_map = {} # 为了方便定位排序后的 solution，使用 map 存储 id 和 solution 的映射关系。
    self.task_trans_metrics = [task["calcMetrics"] for task in self.info["tasks"]] # 任务的输入量。
    self.task_calc_metrics = [self.task_calc_density * task["calcMetrics"] for task in self.info["tasks"]] # 任务的计算量。
    self.calc_abilities = self.get_calc_ability() # node 的计算能力。
    self.link_matrix = np.ones((self.population_size, self.population_size), dtype = int) # 拓扑结构采用随机结构，邻接矩阵[i][j] == 1 表示相邻，[i][j] == 0 表示不相邻。

    for i in range(0, self.population_size):
      self.solutions.append({
        "id": None,
        "HSI": None,
        "vector": np.zeros(self.vector_size, dtype = int),
        "move_in": None,
        "move_out": None,
      })

    self.link() # 形成各个栖息地之间的链接关系。
    # 初始化种群。
    for i in range(0, self.population_size):
      for j in range(0, self.vector_size):
        self.solutions[i]["vector"][j] = random.randint(0, self.node_quantity - 1)
        if self.info["tasks"][j]["calcMetrics"] > self.calc_metrics_threshold:
          self.solutions[i]["vector"][j] = 0 # 将计算量较大的任务初始分配给 node00。  
      self.get_HSI(self.solutions[i])
      self.solutions[i]["id"] = i # 绑定编号和解，之后判断是否链接通过 id 判断。
      self.solutions_map[i] = self.solutions[i]
    
    self.solutions.sort(key=lambda el: el["HSI"], reverse=True) # 为了方便迁移率的计算，按照 HSI 的值降序排序，越靠前，解越差。
    for i in range(0, self.population_size): # 初始化迁移率。
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
  
  def move(self, solution, current_iteration):
    # 进行迁移操作，current_iteration 的取值范围是 [0, self.iterations - 1]。
    new_solution = solution.copy()
    current_maturity = self.maturity_max - ((current_iteration / (self.iterations - 1)) * (self.maturity_max - self.maturity_min)) # 计算成熟度，该值用来计算本次迁移是进行全局迁移还是进行局部迁移。
    for i in range(0, self.vector_size): # 针对解向量中的每个元素。
      if random.random() > solution["move_in"]: continue # 不迁移。
      else: # 执行迁移操作。
        adjacent_move_out_list = [] # 存储相邻栖息地的迁出率。
        adjacent_move_out_id_list = [] # 存储相邻栖息地的 id。
        for j in range(0, len(self.link_matrix[solution["id"]])):
          if self.link_matrix[solution["id"]][j] == 1:
            adjacent_move_out_list.append(self.solutions_map[j]["move_out"])
            adjacent_move_out_id_list.append(j)
        selected_adjacent_solution = self.solutions_map[adjacent_move_out_id_list[roulette(adjacent_move_out_list)]]
        if random.random() > current_maturity:
          # 执行局部迁移。
          solution["vector"][i] = selected_adjacent_solution["vector"][i]
        else:
          # 执行全局迁移。
          non_adjacent_move_out_list = [] # 存储不相邻栖息地的迁出率。
          non_adjacent_move_out_id_list = [] # 存储不相邻栖息地的 id。
          for j in range(0, len(self.link_matrix[solution["id"]])):
            if self.link_matrix[solution["id"]][j] == 0:
              non_adjacent_move_out_list.append(self.solutions_map[j]["move_out"])
              non_adjacent_move_out_id_list.append(j)
          selected_non_adjacent_solution = self.solutions_map[non_adjacent_move_out_id_list[roulette(non_adjacent_move_out_list)]]
          if selected_non_adjacent_solution["HSI"] < selected_adjacent_solution["HSI"]:
            # 不相邻栖息地迁入。
            solution["vector"][i] = selected_non_adjacent_solution["vector"][i]
          else:
            # 相邻栖息地迁入。
            solution["vector"][i] = selected_adjacent_solution["vector"][i]
    self.get_HSI(solution) # 计算新栖息地的 HSI
    if solution["HSI"] < new_solution["HSI"]: return solution
    else: return new_solution
    # 以上为精英保存策略，防止劣化解。
    
    
  def mutation(self):
    print()

  def link(self):
    adjacent_probability = self.neighbor_num / (self.population_size - 1)
    for i in range(0, self.population_size):
      for j in range(0, self.population_size):
        if i == j: continue
        if random.random() < adjacent_probability: 
          self.link_matrix[i][j] = 1
          self.link_matrix[j][i] = 1
        else: 
          self.link_matrix[i][j] = 0
          self.link_matrix[j][i] = 0


if __name__ == "__main__":
  # 读取数据。
  cwd_path = os.getcwd()
  with open(os.path.join(cwd_path, "./BBO/mock.json"), "r") as mock:
    mock_json = json.load(mock)
  BBO(mock_json)