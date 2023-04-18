import sys
sys.dont_write_bytecode = True


from benchmark import ackley_generator, ackley_max, ackley_min
from utils import roulette
import random
import numpy as np
import math
from tqdm import tqdm
import os

class BBO_Pro:
  def __init__(self, target_fn_generator, vector_size, target_fn_max, target_fn_min, solution_size, domain_left, domain_right, iterations):
    # 定义参数。
    self.target_fn = target_fn_generator(vector_size) # 目标函数。
    self.vector_size = vector_size # 作为解的向量维度。
    self.target_fn_max = target_fn_max # 目标函数最大值。
    self.target_fn_min = target_fn_min # 目标函数最小值。
    self.solution_size = solution_size # 种群大小。
    self.move_in_max = 1 # 迁入率的最大值。
    self.move_out_max = 1 # 迁出率的最大值。
    self.neighbor_num = 5 # 每个栖息地的相邻栖息地个数。
    self.maturity_max = 0.75 # 成熟度的最大值。
    self.maturity_min = 0.3 # 成熟度的最小值。
    self.iteration_threshold = 550 # 进入后期变异概率自适应阶段的迭代次数阈值。
    self.mutation_m1 = 0.01 # 前期寻优阶段的固定变异概率。
    self.mutation_m2 = 0.0025 # 后期自适应阶段的基础变异概率。
    self.iterations = iterations # 迭代次数。
    self.domain_left = domain_left # 定义域极左。
    self.domain_right = domain_right # 定义域极右。
    self.ackley_value_sum = 0 # 种群中所有解对应的 ackley value 的和值。
    self.ackley_value_min = sys.maxsize # 群体中 ackley_value 的最小值。 
    self.ackley_value_min_ids = [] # 群体中具有最小 ackley_value 值的 id（可能有多个）。

    # 定义变量。
    self.solutions = [] # 解向量。
    self.solution_id_map = {} # 用于完成 id 和 solution 的映射。
    self.link_matrix = np.zeros((self.solution_size, self.solution_size), dtype = int) # 拓扑结构采用随机结构，邻接矩阵[i][j] == 1 表示相邻，[i][j] == 0 表示不相邻，特别的定义 [i][i] == 0。
    self.data_min = [] # 存储迭代过程中的最优值。

    # 初始化。
    for i in range(0, self.solution_size):
      solution = {
        "id": i, # 给解进行编号。
        "ackley_value": None,
        "vector": [random.randint(self.domain_left, self.domain_right) for _ in range(0, self.vector_size)], # 设置定义域。
        "move_in": None,
        "move_out": None,
      }
      self.solutions.append(solution)
      self.solution_id_map[i] = solution
      self.get_ackley_value(solution)
      self.ackley_value_sum += solution["ackley_value"]
      self.get_ackley_value_min(solution)

    self.link() # 形成各个栖息地之间的链接关系。
    
    self.solutions.sort(key=lambda el: el["ackley_value"], reverse=True)
    # 迭代。
    for i in tqdm(range(0, self.iterations)):
      # 为了方便迁移率的计算，按照 ackley_value 的值降序排序，越靠前，解越差。
      for solution in self.solutions:
        # 计算迁移率。
        self.get_move(solution)
      for solution in self.solutions:
        self.move(solution, i)
        self.mutation(solution, i)
      self.solutions.sort(key=lambda el: el["ackley_value"], reverse=True)
      self.data_min.append([i + 1, self.ackley_value_min])
    
    cwd_path = os.getcwd()
    np.savetxt(os.path.join(cwd_path, "./algorithm/data/BBO-Pro.txt"), np.array(self.data_min), header="Iteration Ackley-Value",  fmt="%d %f")
    self.solutions.sort(key=lambda el: el["ackley_value"])
  
  def get_best_solution(self):
    best_solution = self.solutions[0]["vector"]
    best_ackley_value = self.solutions[0]["ackley_value"]
    return [best_solution, best_ackley_value]
  
  def get_ackley_value_min(self, solution):
    if solution["ackley_value"] < self.ackley_value_min: # 更新 ackley_value_min。
      self.ackley_value_min = solution["ackley_value"]
      self.ackley_value_min_ids.clear()
      self.ackley_value_min_ids.append(solution["id"])
    elif solution["ackley_value"] == self.ackley_value_min:
      self.ackley_value_min_ids.append(solution["id"])
  
  def get_move(self, solution):
    # 计算迁入迁出率。
    solution["move_in"] = (self.move_in_max / 2) * (math.cos(((self.target_fn_max - solution["ackley_value"]) * math.pi) / (self.target_fn_max - self.target_fn_min)) + 1)
    solution["move_out"] = (self.move_out_max / 2) * (-math.cos(((self.target_fn_max - solution["ackley_value"]) * math.pi) / (self.target_fn_max - self.target_fn_min)) + 1)
  
  def get_ackley_value(self, solution):
    solution["ackley_value"] = self.target_fn(solution["vector"])

  def link(self):
    adjacent_probability = self.neighbor_num / (self.solution_size - 1)
    for i in range(0, self.solution_size):
      for j in range(0, self.solution_size):
        if i == j: continue
        if random.random() < adjacent_probability: 
          self.link_matrix[i][j] = 1
          self.link_matrix[j][i] = 1
        else: 
          self.link_matrix[i][j] = 0
          self.link_matrix[j][i] = 0
  
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
          if selected_non_adjacent_solution["ackley_value"] < selected_adjacent_solution["ackley_value"]:
            # 不相邻栖息地迁入。
            solution["vector"][i] = selected_non_adjacent_solution["vector"][i]
          else:
            # 相邻栖息地迁入。
            solution["vector"][i] = selected_adjacent_solution["vector"][i]
    self.get_ackley_value(solution) # 计算新栖息地的 ackley_value
    if solution["ackley_value"] < copy_solution["ackley_value"]: # 得到优化解。
      self.ackley_value_sum -= copy_solution["ackley_value"] # 更新 ackley_value_sum。
      self.ackley_value_sum += solution["ackley_value"]
      self.get_ackley_value_min(solution) # 更新 ackley_value_min。
    else: # 得到劣化解。
      solution["ackley_value"] = copy_solution["ackley_value"]
      solution["vector"] = copy_solution["vector"]
      # 以上为精英保存策略，防止劣化解。

  def mutation(self, solution, current_iteration):
    # 算法的变异阶段分为两部分，前期执行固定概率的变异操作，而后期则使变异概率随着则随着解质量的改变而动态变化。
    mutation_p = None
    if current_iteration <= self.iteration_threshold: # 前期寻优阶段。
      mutation_p = self.mutation_m1
    else: # 后期自适应阶段。
      if (self.ackley_value_sum / self.solution_size) - self.ackley_value_min == 0: return
      mutation_p = self.mutation_m2 * ((solution["ackley_value"] - self.ackley_value_min) / ((self.ackley_value_sum / self.solution_size) - self.ackley_value_min))
    copy_solution = None
    if solution["id"] in self.ackley_value_min_ids and len(self.ackley_value_min_ids) == 1: # 是最优解且最优解只有一个。
      copy_solution = solution.copy()
    for i in range(0, self.vector_size):
      if random.random() < mutation_p:
        # 执行变异操作。
        solution["vector"][i] = random.randint(self.domain_left, self.domain_left)
    if solution["id"] in self.ackley_value_min_ids and len(self.ackley_value_min_ids) == 1:
      # 防止变异破坏最优解。
      self.get_ackley_value(solution)
      if solution["ackley_value"] < copy_solution["ackley_value"]: # 得到优化解。
        self.ackley_value_sum -= copy_solution["ackley_value"] # 更新 ackley_value_sum。
        self.ackley_value_sum += solution["ackley_value"]
        self.get_ackley_value_min(solution) # 更新 ackley_value_min。
      else: # 得到劣化解。
        solution["ackley_value"] = copy_solution["ackley_value"]
        solution["vector"] = copy_solution["vector"] 
    else:
      self.ackley_value_sum -= solution["ackley_value"] # 更新 ackley_value_sum。
      self.get_ackley_value(solution) # 更新变异后的 ackley_value。
      self.ackley_value_sum += solution["ackley_value"]
      self.get_ackley_value_min(solution) # 更新 ackley_value_min。


if __name__ == "__main__":
  # ackley_generator 用于生成 Ackley，但是其需要一个参数用于指定维度。
  vector_size = int(sys.argv[1])
  print() # 空行。
  solution = BBO_Pro(ackley_generator, vector_size, ackley_max, ackley_min, 50, -5, 5, 50)
  print() # 空行。
  [best_solution, best_ackley_value] = solution.get_best_solution()
  print("best_solution: ", best_solution)
  print("best_ackley_value: ", best_ackley_value)