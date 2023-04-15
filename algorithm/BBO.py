import sys
sys.dont_write_bytecode = True

from benchmark import ackley_generator, ackley_max, ackley_min
from utils import roulette
import random

class BBO:
  def __init__(self, target_fn_generator, vector_size, target_fn_max, target_fn_min, solution_size, domain_left, domain_right, iterations):
    
    # 定义参数。
    self.target_fn = target_fn_generator(vector_size) # 目标函数。
    self.vector_size = vector_size # 作为解的向量维度。
    self.target_fn_max = target_fn_max # 目标函数最大值。
    self.target_fn_min = target_fn_min # 目标函数最小值。
    self.solution_size = solution_size # 种群大小。
    self.move_in_max = 1 # 迁入率最大值。
    self.move_out_max = 1 # 迁出率最大值。
    self.mutation_p = 0.01 # 变异率。
    self.iterations = iterations # 迭代次数。
    self.domain_left = domain_left # 定义域极左。
    self.domain_right = domain_right # 定义域极右。

    # 定义变量。
    self.solutions = [] # 解向量。
    self.solution_id_map = {} # 用于完成 id 和 solution 的映射。

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
    
    # 迭代。
    for _ in range(0, self.iterations):
      for solution in self.solutions:
        self.get_move(solution)
      for solution in self.solutions:
        self.move(solution)
        self.mutation(solution)
    
    self.solutions.sort(key=lambda el: el["HSI"])
  
  def get_best_solution(self):
    best_solution = self.solutions[0]["vector"]
    best_HSI = self.solutions[0]["HSI"]
    print(best_solution, best_HSI)


  def get_move(self, solution):
    solution["move_in"] = self.move_in_max * ((self.target_fn_max - solution["HSI"]) / (self.target_fn_max - self.target_fn_min))
    solution["move_out"] = self.move_out_max * ((solution["HSI"] - self.target_fn_min) / (self.target_fn_max - self.target_fn_min))
  
  def get_HSI(self, solution):
    # 计算适应度。
    solution["HSI"] = self.target_fn(solution["vector"])

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
    for i in range(0, self.vector_size):
      if random.random() < self.mutation_p:
        solution["vector"][i] = random.randint(self.domain_left, self.domain_right)
    self.get_HSI(solution)

if __name__ == "__main__":
  # ackley_generator 用于生成 Ackley，但是其需要一个参数用于指定维度。
  solution = BBO(ackley_generator, 6, ackley_max, ackley_min, 50, -5, 5, 10000)
  solution.get_best_solution()

