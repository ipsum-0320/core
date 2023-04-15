import sys
sys.dont_write_bytecode = True


from benchmark import ackley_generator, ackley_max, ackley_min
from utils import roulette
import random
import numpy as np
import math


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
    self.HSI_sum = 0 # 群体 HSI 的均值。
    self.HSI_min = sys.maxsize # 群体中 HSI 的最小值。 
    self.HSI_min_ids = [] # 群体中具有最小 HSI 值的 id（可能有多个）。

    # 定义变量。
    self.solutions = [] # 解向量。
    self.solution_id_map = {} # 用于完成 id 和 solution 的映射。
    self.link_matrix = np.zeros((self.solution_size, self.solution_size), dtype = int) # 拓扑结构采用随机结构，邻接矩阵[i][j] == 1 表示相邻，[i][j] == 0 表示不相邻，特别的定义 [i][i] == 0。

    # 初始化。
    for i in range(0, self.solution_size):
      solution = {
        "id": i, # 给解进行编号。
        "HSI": None,
        "vector": [random.randint(self.domain_left, self.domain_right) for _ in range(0, self.vector_size)], # 设置定义域。
        "move_in": None,
        "move_out": None,
      }
      self.solutions.append(solution)
      self.solution_id_map[i] = solution
      self.get_HSI(solution)
      self.HSI_sum += solution["HSI"]
      self.get_HSI_min(solution)

    self.link() # 形成各个栖息地之间的链接关系。
    
    # 迭代。
    for i in range(0, self.iterations):
      self.solutions.sort(key=lambda el: el["HSI"], reverse=True) 
      # 为了方便迁移率的计算，按照 HSI 的值降序排序，越靠前，解越差。
      for solution in self.solutions:
        # 计算迁移率。
        self.get_move(solution)
      for solution in self.solutions:
        self.move(solution, i)
        self.mutation(solution, i)
    
    self.solutions.sort(key=lambda el: el["HSI"])
  
  def get_best_solution(self):
    best_solution = self.solutions[0]["vector"]
    best_HSI = self.solutions[0]["HSI"]
    print(best_solution, best_HSI)
  
  def get_HSI_min(self, solution):
    if solution["HSI"] < self.HSI_min: # 更新 HSI_min。
      self.HSI_min = solution["HSI"]
      self.HSI_min_ids.clear()
      self.HSI_min_ids.append(solution["id"])
    elif solution["HSI"] == self.HSI_min:
      self.HSI_min_ids.append(solution["id"])
  
  def get_move(self, solution):
    # 计算迁入迁出率。
    solution["move_in"] = (self.move_in_max / 2) * (math.cos(((self.target_fn_max - solution["HSI"]) * math.pi) / (self.target_fn_max - self.target_fn_min)) + 1)
    solution["move_out"] = (self.move_out_max / 2) * (-math.cos(((self.target_fn_max - solution["HSI"]) * math.pi) / (self.target_fn_max - self.target_fn_min)) + 1)
  
  def get_HSI(self, solution):
    # 计算适应度。
    solution["HSI"] = self.target_fn(solution["vector"])

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
    for i in range(0, self.vector_size):
      if random.random() < mutation_p:
        # 执行变异操作。
        solution["vector"][i] = random.randint(self.domain_left, self.domain_left)
    self.HSI_sum -= solution["HSI"] # 更新 HSI_sum。
    self.get_HSI(solution) # 更新变异后的 HSI。
    self.HSI_sum += solution["HSI"]
    self.get_HSI_min(solution) # 更新 HSI_min。


if __name__ == "__main__":
  # ackley_generator 用于生成 Ackley，但是其需要一个参数用于指定维度。
  solution = BBO_Pro(ackley_generator, 10, ackley_max, ackley_min, 50, -5, 5, 10000)
  solution.get_best_solution()