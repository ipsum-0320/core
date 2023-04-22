import sys
sys.dont_write_bytecode = True

import json
import random
import os
import math

from typing import TypedDict, List
import numpy as np
from decimal import Decimal, getcontext
from tqdm import tqdm


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

class BBO_Pro:
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
          "podName": "%s-%d" % (val["podName"][:-4], i),
          "image": val["image"],
          "calcMetrics": int(val["calcMetrics"])
        }
        self.info["tasks"].append(task)
    
    # 定义参数。
    self.iterations = 150 # 算法的迭代次数。
    self.solution_size = 50 # 种群中栖息地的数量，这里指代解的数量。
    self.S_max = 50 # 种群数量的最大值，用于计算迁入迁出率，为了能够同时去到 0 和 1，取 S_max = solution_size - 1。
    self.vector_size = len(self.info["tasks"]) # 解向量的长度。
    self.node_quantity = 6 # 可用于运行 pod 的工作节点数量。
    self.calc_metrics_threshold = 300 # 初始化时，计算任务被分配到 node00 上的计算量阈值。
    self.time_weight = [round(0.85 + x / 100, 2) for x in range(0, 6)] # 为了能够尽可能取得最优解，选择多个权向量确定不同的搜索方向。
    self.logistics_K = 0.000025 # logistics 函数中的 K 值，参数的确定基于函数图像的调整。
    self.logistics_X_0 = 300000 # logistics 函数中的 X_0 值，参数的确定基于函数图像的调整。
    self.task_calc_density = 23 # 任务的计算密度。
    self.trans_bandwidth = 20.29 # 传输带宽。
    self.energy_density = 11.3453 # 任务的能量密度。
    self.e_per_time = self.task_calc_density * self.trans_bandwidth # 单位时间的传输所消耗的能量。
    self.move_in_max = 1 # 迁入率的最大值。
    self.move_out_max = 1 # 迁出率的最大值。
    self.neighbor_num = 5 # 每个栖息地的相邻栖息地个数。
    self.maturity_max = 0.75 # 成熟度的最大值。
    self.maturity_min = 0.3 # 成熟度的最小值。
    self.iteration_threshold = 550 # 进入后期变异概率自适应阶段的迭代次数阈值。
    self.mutation_m1 = 0.01 # 前期寻优阶段的固定变异概率。
    self.mutation_m2 = 0.0025 # 后期自适应阶段的基础变异概率。
    self.HSI_sum = 0 # 群体 HSI 的和值。
    self.HSI_min = sys.maxsize # 群体中 HSI 的最小值。 
    self.HSI_min_ids = [] # 群体中具有最小 HSI 值的 id（可能有多个）。
    self.mutation_to_node00_list_len = 8 # 变异为 node00 的随机数列表长度。

    # 定义变量。
    self.solutions = [] # 一个复杂的数据结构，包含 id、解、迁入迁出率和 HSI 值。
    self.solution_id_map = {} # 为了方便定位排序后的 solution，使用 map 存储 id 和 solution 的映射关系。
    self.task_trans_metrics = [task["calcMetrics"] for task in self.info["tasks"]] # 任务的输入量。
    self.task_calc_metrics = [self.task_calc_density * task["calcMetrics"] for task in self.info["tasks"]] # 任务的计算量。
    self.calc_abilities = self.get_calc_ability() # node 的计算能力。
    self.link_matrix = np.zeros((self.solution_size, self.solution_size), dtype = int) # 拓扑结构采用随机结构，邻接矩阵[i][j] == 1 表示相邻，[i][j] == 0 表示不相邻。
    self.data_min = [] # 存储迭代过程中的最优值。

    # 用于归一化的参数。
    self.min_T = self.get_min_T() # T 的最小值，用于归一化。
    self.max_T = self.get_max_T() # T 的最大值，用于归一化。
    self.min_E = sum([self.energy_density * calc for calc in self.task_calc_metrics]) # E 的最小值，用于归一化。
    self.max_E = self.min_E + sum([self.e_per_time * tran / self.trans_bandwidth for tran in self.task_trans_metrics]) # E 的最大值，用于归一化。

    for i in range(0, self.solution_size):
      self.solutions.append({
        "id": None,
        "HSI": None,
        "vector": np.zeros(self.vector_size, dtype = int),
        "move_in": None,
        "move_out": None,
      })

    # 初始化种群。
    for i in range(0, self.solution_size):
      for j in range(0, self.vector_size):
        self.solutions[i]["vector"][j] = random.randint(0, 5)
        if self.info["tasks"][j]["calcMetrics"] > self.calc_metrics_threshold:
          self.solutions[i]["vector"][j] = 0 # 将计算量较大的任务初始分配给 node00（输出保证 0 号元素就是 node00）。
      self.get_HSI(self.solutions[i])
      self.solutions[i]["id"] = i # 绑定编号和解，之后判断是否链接通过 id 判断。
      self.solution_id_map[i] = self.solutions[i]
      self.HSI_sum += self.solutions[i]["HSI"]
      self.get_HSI_min(self.solutions[i])
      
    self.link() # 形成各个栖息地之间的链接关系。
    
    # 迭代。
    self.solutions.sort(key=lambda el: el["HSI"])
    for i in tqdm(range(0, self.iterations)):
      # 为了方便迁移率的计算，按照 HSI 的值升序排序，越靠前，解越好。
      for j in range(0, self.solution_size):
        # 计算迁移率。
        self.get_move(self.solutions[j], j)
      for j in range(0, len(self.solutions)):
        self.move(self.solutions[j], i)
        self.mutation(self.solutions[j], i)
      self.solutions.sort(key=lambda el: el["HSI"])
      self.data_min.append([i + 1, self.HSI_min])

    cwd_path = os.getcwd()
    np.savetxt(os.path.join(cwd_path, "./application/data/BBO-Pro.txt"), np.array(self.data_min), header="Iteration Cost",  fmt="%d %f")
    self.solutions.sort(key=lambda el: el["HSI"]) # 按照 HSI 的值升序排序，排序最前的即是最优解。
    
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
  
  def get_calc_ability(self):
    # 计算 node 的计算能力。
    calc_abilities = []
    for node in self.info["nodes"]:
      calc_abilities.append(round((1 / (1 + (math.e ** -(self.logistics_K * (node["mem"] - self.logistics_X_0))))) * node["cpu"], 2))
      # 使用 logistics 函数计算。
    return calc_abilities
  
  def move(self, solution, current_iteration):
    # 进行迁移操作，current_iteration 的取值范围是 [0, self.iterations - 1]。
    copy_solution = solution.copy()
    current_maturity = self.maturity_max - ((current_iteration / (self.iterations - 1)) * (self.maturity_max - self.maturity_min)) # 计算成熟度，该值用来计算本次迁移是进行全局迁移还是进行局部迁移。
    for i in range(0, self.vector_size): # 针对解向量中的每个元素。
      if random.random() >= solution["move_in"]: continue # 不迁移。
      else: # 执行迁移操作。
        adjacent_move_out_list = [] # 存储相邻栖息地的迁出率。
        adjacent_move_out_id_list = [] # 存储相邻栖息地的 id。
        for j in range(0, len(self.link_matrix[solution["id"]])):
          if self.link_matrix[solution["id"]][j] == 1:
            adjacent_move_out_list.append(self.solution_id_map[j]["move_out"])
            adjacent_move_out_id_list.append(j)
        if len(adjacent_move_out_list) == 0:
          # 没有相邻的栖息地，直接执行全局迁移。
          non_adjacent_move_out_list = [] # 存储不相邻栖息地的迁出率。
          non_adjacent_move_out_id_list = [] # 存储不相邻栖息地的 id。
          for j in range(0, len(self.link_matrix[solution["id"]])):
            if self.link_matrix[solution["id"]][j] == 0:
              if solution["id"] == j: continue
              non_adjacent_move_out_list.append(self.solution_id_map[j]["move_out"])
              non_adjacent_move_out_id_list.append(j)
          selected_non_adjacent_solution = self.solution_id_map[non_adjacent_move_out_id_list[roulette(non_adjacent_move_out_list)]]
          solution["vector"][i] = selected_non_adjacent_solution["vector"][i]
          continue 
        selected_adjacent_solution = self.solution_id_map[adjacent_move_out_id_list[roulette(adjacent_move_out_list)]]
        if random.random() > current_maturity:
          # 执行局部迁移。
          solution["vector"][i] = selected_adjacent_solution["vector"][i]
        else:
          # 执行全局迁移。
          non_adjacent_move_out_list = [] # 存储不相邻栖息地的迁出率。
          non_adjacent_move_out_id_list = [] # 存储不相邻栖息地的 id。
          for j in range(0, len(self.link_matrix[solution["id"]])):
            if self.link_matrix[solution["id"]][j] == 0:
              if solution["id"] == j: continue
              non_adjacent_move_out_list.append(self.solution_id_map[j]["move_out"])
              non_adjacent_move_out_id_list.append(j)
          if len(non_adjacent_move_out_list) == 0:
            # 栖息地全部连接，直接从栖息地中执行迁移。
            solution["vector"][i] = selected_adjacent_solution["vector"][i]
            continue
          selected_non_adjacent_solution = self.solution_id_map[non_adjacent_move_out_id_list[roulette(non_adjacent_move_out_list)]]
          if selected_non_adjacent_solution["HSI"] < selected_adjacent_solution["HSI"]:
            # 不相邻栖息地迁入。
            solution["vector"][i] = selected_non_adjacent_solution["vector"][i]
          else:
            # 相邻栖息地迁入。
            solution["vector"][i] = selected_adjacent_solution["vector"][i]
    self.get_HSI(solution) # 计算新栖息地的 HSI
    if solution["HSI"] < copy_solution["HSI"]: # 得到优化解。
      self.HSI_sum -= copy_solution["HSI"] # 更新 HSI_sum。
      self.HSI_sum += solution["HSI"]
      self.get_HSI_min(solution) # 更新 HSI_min。
    else: # 得到劣化解。
      solution["HSI"] = copy_solution["HSI"]
      solution["vector"] = copy_solution["vector"]
      # 以上为精英保存策略，防止劣化解。
    
  def mutation(self, solution, current_iteration):
    # 算法的变异阶段分为两部分，前期执行固定概率的变异操作，而后期则使变异概率随着则随着解质量的改变而动态变化。
    mutation_p = None
    if current_iteration <= self.iteration_threshold: # 前期寻优阶段。
      mutation_p = self.mutation_m1
    else: # 后期自适应阶段。
      mutation_p = self.mutation_m2 * ((solution["HSI"] - self.HSI_min) / ((self.HSI_sum / self.solution_size) - self.HSI_min))
    copy_solution = None
    if solution["id"] in self.HSI_min_ids and len(self.HSI_min_ids) == 1: # 是最优解且最优解只有一个。
      copy_solution = solution.copy()
    for i in range(0, self.vector_size):
      if random.random() < mutation_p:
        # 执行变异操作。
        node = random.randint(-(self.mutation_to_node00_list_len - 1), self.node_quantity - 1) 
        # 如果 node 取值为 [-(mutation_to_node00_list_len - 1), ..., 0]，则需要调度到云中心。
        solution["vector"][i] = 0 if node <= 0 else node
        # 由于云中心的计算能力占比较大，因此调度去云中心的概率应该较高。
        # 在这里，我们选择使用调整 random.randint 取值，来增大其调度去云中心的概率。
    if solution["id"] in self.HSI_min_ids and len(self.HSI_min_ids) == 1:
      # 防止变异破坏最优解。
      self.get_HSI(solution)
      if solution["HSI"] < copy_solution["HSI"]: # 得到优化解。
        self.HSI_sum -= copy_solution["HSI"] # 更新 HSI_sum。
        self.HSI_sum += solution["HSI"]
        self.get_HSI_min(solution) # 更新 HSI_min。
      else: # 得到劣化解。
        solution["HSI"] = copy_solution["HSI"]
        solution["vector"] = copy_solution["vector"]      
    else:
      self.HSI_sum -= solution["HSI"] # 更新 HSI_sum。
      self.get_HSI(solution) # 更新变异后的 HSI。
      self.HSI_sum += solution["HSI"]
      self.get_HSI_min(solution) # 更新 HSI_min。

  def link(self):
    adjacent_probability = self.neighbor_num / (self.solution_size - 1)
    for i in range(0, self.solution_size):
      for j in range(0, self.solution_size):
        if i == j: continue
        if random.random() < adjacent_probability: 
          self.link_matrix[i][j] = 1
          self.link_matrix[j][i] = 1

  def get_HSI_min(self, solution):
    if solution["HSI"] < self.HSI_min: # 更新 HSI_min。
      self.HSI_min = solution["HSI"]
      self.HSI_min_ids.clear()
      self.HSI_min_ids.append(solution["id"])
    elif solution["HSI"] == self.HSI_min:
      self.HSI_min_ids.append(solution["id"])

if __name__ == "__main__":
  # 读取数据。
  cwd_path = os.getcwd()
  with open(os.path.join(cwd_path, "./application/mock.json"), "r") as mock:
    mock_json = json.load(mock)
  print() # 空行。
  solution = BBO_Pro(mock_json)
  print() # 空行。
  for s in solution.get_solution():
    print(s)