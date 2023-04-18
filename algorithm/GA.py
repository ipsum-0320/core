import sys
sys.dont_write_bytecode = True

from benchmark import ackley_generator, ackley_max, ackley_min
from utils import roulette
import random
from tqdm import tqdm
import numpy as np
import os
import math
import uuid

class GA:
  def __init__(self, target_fn_generator, vector_size, target_fn_max, target_fn_min, solution_size, domain_left, domain_right, iterations):
    # 定义参数。
    self.target_fn = target_fn_generator(vector_size) # 目标函数。
    self.vector_size = vector_size # 作为解的向量维度。
    self.target_fn_max = target_fn_max # 目标函数最大值。
    self.target_fn_min = target_fn_min # 目标函数最小值。
    self.solution_size = solution_size # 种群大小。
    self.selected_solution_nums = math.floor(solution_size / 2) # 被选中的父母解的个数。
    self.iterations = iterations # 迭代次数。
    self.domain_left = domain_left # 定义域极左。
    self.domain_right = domain_right # 定义域极右。
    self.crossover_p = 0.75 # 父母个体染色体的交叉概率，一般取值范围在 [0.6, 0.9]。
    self.mutation_p = 0.05 # 后代染色体的变异概率，一般取值范围在 [0.01, 0.1]。
    self.bin_len = len(bin(abs(domain_left) if abs(domain_left) >= abs(domain_right) else abs(domain_right))[2:]) # 二进制编码的长度，注意此处不包含符号位。
    self.domain_interval_len = domain_right - domain_left + 1 # 定义域的区间长度。

    # 定义变量。
    self.solutions = [] # 解向量。
    self.solution_id_map = {} # 用于完成 id 和 solution 的映射。
    self.data_min = [] # 存储迭代过程中的最优值。
    self.selected_solutions = [] # 被选中的父母解。
    self.selected_solutions_not_crossovered = {} # 被选中且尚未交叉的父母解。
    self.offspring_solutions = [] # 后代解向量。

    # 初始化。
    for i in range(0, self.solution_size):
      solution = {
        "id": uuid.uuid1(), # 给解进行编号。
        "ackley_value": None,
        "vector": [random.randint(self.domain_left, self.domain_right) for _ in range(0, self.vector_size)], # 设置定义域。
      }
      self.solutions.append(solution)
      self.get_ackley_value(solution)
    
    # 迭代。
    for i in tqdm(range(0, self.iterations)):
      self.select()
      for solution in self.selected_solutions:
        if solution["is_crossover"] == False:
          self.crossover(solution)
      self.solutions = self.solutions + self.offspring_solutions # 合并种群。
      self.solutions.sort(key=lambda el: el["ackley_value"]) # 筛选优质个体。
      self.solutions = self.solutions[:self.solution_size]
      self.data_min.append([i + 1, self.solutions[0]["ackley_value"]])

    cwd_path = os.getcwd()
    np.savetxt(os.path.join(cwd_path, "./algorithm/data/GA.txt"), np.array(self.data_min), header="Iteration Ackley-Value",  fmt="%d %f")

  def get_best_solution(self):
    best_solution = self.solutions[0]["vector"]
    best_ackley_value = self.solutions[0]["ackley_value"]
    return [best_solution, best_ackley_value]

  def get_ackley_value(self, solution):
    # 计算 ackley value 的取值。
    solution["ackley_value"] = self.target_fn(solution["vector"])
  
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
      if value > self.domain_right:
        value = self.domain_left + ((value - self.domain_left) % self.domain_interval_len)
      elif value < self.domain_left:
        value = self.domain_right - (abs(value - self.domain_right) % self.domain_interval_len)
      vector.append(value)
    return vector

  
  def select(self):
    # 选择。
    self.solution_id_map.clear() # 实现 id 和 solution 的映射。
    for solution in self.solutions:
      self.solution_id_map[solution["id"]] = solution
    self.selected_solutions.clear()
    self.selected_solutions_not_crossovered.clear()
    self.offspring_solutions.clear()
    ackley_value_id_map = {} # id 和 ackley value 的映射。
    for i in range(0, len(self.solutions)):
      ackley_value_id_map[self.solutions[i]["id"]] = self.solutions[i]["ackley_value"]
    for _ in range(0, self.selected_solution_nums):
      ackley_value_list = [v for _, v in ackley_value_id_map.items()]
      id_list = [k for k, _ in ackley_value_id_map.items()]
      target_index = roulette(ackley_value_list)
      target_solution = self.solution_id_map[id_list[target_index]]
      selected_solution = {
        "id": uuid.uuid1(), 
        "ackley_value": target_solution["ackley_value"],
        "vector": target_solution["vector"],
        "is_crossover": False # 是否被交叉过。 
      }
      self.selected_solutions.append(selected_solution)
      self.selected_solutions_not_crossovered[selected_solution["id"]] = selected_solution
      del ackley_value_id_map[id_list[target_index]]
    
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
      # 更新 ackley value。
      self.get_ackley_value(solution)
      self.get_ackley_value(target_solution)
      # 添加。
      self.offspring_solutions.append(solution)
      self.offspring_solutions.append(target_solution)
    else:
      # 不交叉，本身作为子代。
      mutationed_vector_bin = self.mutation(vector_bin)
      solution["vector"] = self.decode(mutationed_vector_bin)
      self.get_ackley_value(solution)
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
  # ackley_generator 用于生成 Ackley，但是其需要一个参数用于指定维度。
  vector_size = int(sys.argv[1])
  print() # 空行。
  solution = GA(ackley_generator, vector_size, ackley_max, ackley_min, 50, -5, 5, 50)
  print() # 空行。
  [best_solution, best_ackley_value] = solution.get_best_solution()
  print("best_solution: ", best_solution)
  print("best_ackley_value: ", best_ackley_value)
