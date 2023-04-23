import sys
sys.dont_write_bytecode = True

import os
import json
import random
import math
from typing import TypedDict, List
import uuid
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

class GA:
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
    
    # 参数设置。
    self.iterations = 80 # 算法的迭代次数。
    self.solution_size = 50 # 种群中栖息地的数量，这里指代解的数量。
    self.vector_size = len(self.info["tasks"]) # 解向量的长度。
    self.node_quantity = 6 # 可用于运行 pod 的工作节点数量。
    self.time_weight = [round(0.85 + x / 100, 2) for x in range(0, 6)] # 为了能够尽可能取得最优解，选择多个权向量确定不同的搜索方向。
    self.logistics_K = 0.000025 # logistics 函数中的 K 值，参数的确定基于函数图像的调整。
    self.logistics_X_0 = 300000 # logistics 函数中的 X_0 值，参数的确定基于函数图像的调整。
    self.task_calc_density = 23 # 任务的计算密度。
    self.trans_bandwidth = 20.29 # 传输带宽。
    self.energy_density = 11.3453 # 任务的能量密度。
    self.e_per_time = self.task_calc_density * self.trans_bandwidth # 单位时间的传输所消耗的能量。
    self.crossover_p = 0.75 # 父母个体染色体的交叉概率，一般取值范围在 [0.6, 0.9]。
    self.mutation_p = 0.05 # 后代染色体的变异概率，一般取值范围在 [0.01, 0.1]。
    self.bin_len = len(bin(5)[2:]) # 二进制编码的长度，注意此处不包含符号位。
    self.domain_interval_len = 6 # 定义域的区间长度。
    self.selected_solution_nums = 25 # 被选中的父母解的个数。


    # 定义变量。
    self.solutions = [] # 解向量。
    self.solution_id_map = {} # 为了方便定位排序后的 solution，使用 map 存储 id 和 solution 的映射关系。
    self.task_trans_metrics = [task["calcMetrics"] for task in self.info["tasks"]] # 任务的输入量。
    self.task_calc_metrics = [self.task_calc_density * task["calcMetrics"] for task in self.info["tasks"]] # 任务的计算量。
    self.calc_abilities = self.get_calc_ability() # node 的计算能力。
    self.selected_solutions = [] # 被选中的父母解。
    self.selected_solutions_not_crossovered = {} # 被选中且尚未交叉的父母解。
    self.offspring_solutions = [] # 后代解向量。
    self.data_min = [] # 存储迭代过程中的最优值。

    # 用于归一化的参数。
    self.min_T = self.get_min_T() # T 的最小值，用于归一化。
    self.max_T = self.get_max_T() # T 的最大值，用于归一化。
    self.min_E = sum([self.energy_density * calc for calc in self.task_calc_metrics]) # E 的最小值，用于归一化。
    self.max_E = self.min_E + sum([self.e_per_time * tran / self.trans_bandwidth for tran in self.task_trans_metrics]) # E 的最大值，用于归一化。

    # 初始化。
    for i in range(0, self.solution_size):
      solution = {
        "id": uuid.uuid1(), # 给解进行编号。
        "cost": None,
        "vector": [random.randint(0, 5) for _ in range(0, self.vector_size)], # 设置定义域。
      }
      self.solutions.append(solution)
      self.get_cost(solution)

    # 迭代。
    for i in tqdm(range(0, self.iterations)):
      self.select()
      for solution in self.selected_solutions:
        if solution["is_crossover"] == False:
          self.crossover(solution)
      self.solutions = self.solutions + self.offspring_solutions # 合并种群。
      self.solutions.sort(key=lambda el: el["cost"]) # 筛选优质个体。
      self.solutions = self.solutions[:self.solution_size]
      self.data_min.append([i + 1, self.solutions[0]["cost"]])

    cwd_path = os.getcwd()
    np.savetxt(os.path.join(cwd_path, "./application/data/GA.txt"), np.array(self.data_min), header="Iteration Cost",  fmt="%d %f")
  
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

  def get_solution(self):
    best_solution = self.solutions[0]["vector"]
    print("best_cost: ", self.solutions[0]["cost"])
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
  
  def encode(self, vector):
    # 编码。
    vector_bin = ""
    for el in vector:
      el_bin = ""
      if el >= 0: # 非负数补码。
        symbol_bin = "0" # 非负数符号位。
        el_bin = bin(el)[2:].zfill(self.bin_len)
        el_bin = symbol_bin + el_bin # 最终的补码。
      else: # 负数补码。
        symbol_bin = "1" # 负数符号位。
        el_bin_part = bin(abs(el))[2:].zfill(self.bin_len) # 将十进制转化为相应长度的二进制字符串。
        el_bin_part = "".join("1" if c == "0" else "0" for c in el_bin_part) # 取得反码。
        key_index = len(el_bin_part) - 1
        carry = 1
        for _ in range(0, len(el_bin_part)):
          current_bit = int(el_bin_part[key_index]) + carry
          carry = 0
          if current_bit == 2:
            carry = 1
            el_bin = "0" + el_bin
          else:
            el_bin = el_bin_part[:key_index] + "1" + el_bin
            break
          key_index = key_index - 1
        el_bin = symbol_bin + el_bin # 最终的补码。
      vector_bin += el_bin # 串行编码。
    return vector_bin
  
  def decode(self, vector_bin):
    # 解码
    step = self.bin_len + 1
    vector_bin_list = [vector_bin[i:(i + step)] for i in range(0, len(vector_bin), step)]
    vector = []
    for el in vector_bin_list:
      symbol = el[0]
      el = el[1:]
      if symbol == '1': # 负数
        el = "".join("1" if c == "0" else "0" for c in el)
        value = -(int(el, 2) + 1)
      else: # 正数 
        value = int(el, 2)
      # 进行循环映射。
      if value > 5:
        value = value % self.domain_interval_len
      elif value < 0:
        value = 5 - (abs(value - 5) % self.domain_interval_len)
      vector.append(value)
    return vector

  def get_cost(self, solution):
    # 计算 cost，cost 值越小，那么解越优质。
    cost = sys.maxsize
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
      cost = min(cost, Decimal(weight) * normalized_T + Decimal(1 - weight) * normalized_E)
    solution["cost"] = cost

  def select(self):
    # 选择。
    self.solution_id_map.clear() # 实现 id 和 solution 的映射。
    for solution in self.solutions:
      self.solution_id_map[solution["id"]] = solution
    self.selected_solutions.clear()
    self.selected_solutions_not_crossovered.clear()
    self.offspring_solutions.clear()
    cost_id_map = {} # id 和  cost 的映射。
    for i in range(0, len(self.solutions)):
      cost_id_map[self.solutions[i]["id"]] = self.solutions[i]["cost"]
    for _ in range(0, self.selected_solution_nums):
      cost_list = [v for _, v in cost_id_map.items()]
      id_list = [k for k, _ in cost_id_map.items()]
      target_index = roulette(cost_list)
      target_solution = self.solution_id_map[id_list[target_index]]
      selected_solution = {
        "id": uuid.uuid1(), 
        "cost": target_solution["cost"],
        "vector": target_solution["vector"],
        "is_crossover": False # 是否被交叉过。 
      }
      self.selected_solutions.append(selected_solution)
      self.selected_solutions_not_crossovered[selected_solution["id"]] = selected_solution
      del cost_id_map[id_list[target_index]]
  
  def crossover(self, solution):
    # 交叉，参数中的 solution 来自于 self.selected_solutions，是全新的 solution，不会影响 self.solutions。
    solution["is_crossover"] = True
    del self.selected_solutions_not_crossovered[solution["id"]]
    vector_bin = self.encode(solution["vector"])
    if random.random() <= self.crossover_p and len(self.selected_solutions_not_crossovered) > 0:
      # 交叉，交叉结果作为子代。
      target_solution_id = random.choice([k for k, _ in self.selected_solutions_not_crossovered.items()])
      target_solution = self.selected_solutions_not_crossovered[target_solution_id]
      target_solution["is_crossover"] = True
      del self.selected_solutions_not_crossovered[target_solution["id"]]
      target_solution_vector_bin = self.encode(target_solution["vector"])
      # 交叉方式为单点交叉。
      random_index = random.randint(1, self.vector_size * (self.bin_len + 1) - 2)
      offspring_vector_bin = vector_bin[0:random_index] + target_solution_vector_bin[random_index:]
      offspring_target_solution_vector_bin = target_solution_vector_bin[0:random_index] + vector_bin[random_index:]
      # 子代变异。
      mutationed_offspring_vector_bin = self.mutation(offspring_vector_bin)
      mutationed_offspring_target_solution_vector_bin = self.mutation(offspring_target_solution_vector_bin)
      solution["vector"] = self.decode(mutationed_offspring_vector_bin)
      target_solution["vector"] = self.decode(mutationed_offspring_target_solution_vector_bin)
      # 更新 cost。
      self.get_cost(solution)
      self.get_cost(target_solution)
      # 添加。
      self.offspring_solutions.append(solution)
      self.offspring_solutions.append(target_solution)
    else:
      # 不交叉，本身作为子代。
      mutationed_vector_bin = self.mutation(vector_bin)
      solution["vector"] = self.decode(mutationed_vector_bin)
      self.get_cost(solution)
      self.offspring_solutions.append(solution)

  def mutation(self, vector_bin):
    # 变异。
    mutationed_vector_bin = ""
    for el in vector_bin:
      if random.random() < self.mutation_p:
        mutationed_vector_bin += "1" if el == "0" else "0"
      else:
        mutationed_vector_bin += el
    return mutationed_vector_bin

if __name__ == "__main__":
  cwd_path = os.getcwd()
  with open(os.path.join(cwd_path, "./application/mock.json"), "r") as mock:
    mock_json = json.load(mock)
  print() # 空行。
  solution = GA(mock_json)
  print() # 空行。
  for s in solution.get_solution():
    print(s)