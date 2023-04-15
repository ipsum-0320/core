import sys
sys.dont_write_bytecode = True

import random

# 轮盘赌选择算法。
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