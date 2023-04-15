import sys
sys.dont_write_bytecode = True

from benchmark import ackley_generator, ackley_max, ackley_min

class GA:
  def __init__(self, target_fn_generator, vector_size, target_fn_max, target_fn_min, solution_size, domain_left, domain_right, iterations):
    
    print()

if __name__ == "__main__":
  # ackley_generator 用于生成 Ackley，但是其需要一个参数用于指定维度。
  solution = GA(ackley_generator, 6, ackley_max, ackley_min, 50, -5, 5, 10000)
  solution.get_best_solution()