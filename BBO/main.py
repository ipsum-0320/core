import json
import os

from typing import TypedDict, List

class Node:
  nodeName: str
  cpu: float
  mem: float

class Task:
  podName: str
  image: str
  nums: int
  calcMetrics: str

class Info(TypedDict):
  nodes: List[Node]
  tasks: List[Task]

if __name__ == "__main__":
  # 读取数据
  cwd_path = os.getcwd()
  with open(os.path.join(cwd_path, "./BBO/mock.json"), "r") as mock:
    mock_json = json.load(mock)
  info: Info = {
    "nodes": None,
    "tasks": [],
  }
  info["nodes"] = mock_json["nodes"]
  for val in mock_json["tasks"]:
    for i in range(1, int(val["nums"]) + 1):
      task: Task = {
        "podName": "%s-%d" % (val["podName"], i),
        "image": val["image"],
        "calcMetrics": val["calcMetrics"]
      }
      info["tasks"].append(task)
  
  # 初始化种群






