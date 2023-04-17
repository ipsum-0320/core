PYTHON=python
PIP=pip

# 参数组 1 
VECTOR_SIZE_1=5 # 维度较低。
SOLUTION_SIZE_1=50
DOMAIN_LEFT_1=-10
DOMAIN_RIGHT_1=10
ITERATIONS_1=50

# 参数组 2
VECTOR_SIZE_2=20 # 维度适中。
SOLUTION_SIZE_2=50
DOMAIN_LEFT_2=-50
DOMAIN_RIGHT_2=50
ITERATIONS_2=50

# 参数组 3
VECTOR_SIZE_3=60 # 维度较高。
SOLUTION_SIZE_3=50
DOMAIN_LEFT_3=-50
DOMAIN_RIGHT_3=50
ITERATIONS_3=50

MAKEFLAGS += --always-make # 避免使用缓存。

all: BBO-Pro BBO GA

install: requirements.txt
	$(PIP) install -r $^

depend: requirements.txt
	$(PIP) freeze > $^

BBO-Pro:
ifeq ($(p), 1)
	$(PYTHON) algorithm/BBO-Pro.py $(VECTOR_SIZE_1) $(SOLUTION_SIZE_1) $(DOMAIN_LEFT_1) $(DOMAIN_RIGHT_1) $(ITERATIONS_1)
else ifeq ($(p), 2)
	$(PYTHON) algorithm/BBO-Pro.py $(VECTOR_SIZE_2) $(SOLUTION_SIZE_2) $(DOMAIN_LEFT_2) $(DOMAIN_RIGHT_2) $(ITERATIONS_2)
else ifeq ($(p), 3)
	$(PYTHON) algorithm/BBO-Pro.py $(VECTOR_SIZE_3) $(SOLUTION_SIZE_3) $(DOMAIN_LEFT_3) $(DOMAIN_RIGHT_3) $(ITERATIONS_3)
else
	@echo "Please specify a valid set of parameters: p=1/2/3"
endif


BBO: 
ifeq ($(p), 1)
	$(PYTHON) algorithm/BBO.py $(VECTOR_SIZE_1) $(SOLUTION_SIZE_1) $(DOMAIN_LEFT_1) $(DOMAIN_RIGHT_1) $(ITERATIONS_1)
else ifeq ($(p), 2)
	$(PYTHON) algorithm/BBO.py $(VECTOR_SIZE_2) $(SOLUTION_SIZE_2) $(DOMAIN_LEFT_2) $(DOMAIN_RIGHT_2) $(ITERATIONS_2)
else ifeq ($(p), 3)
	$(PYTHON) algorithm/BBO.py $(VECTOR_SIZE_3) $(SOLUTION_SIZE_3) $(DOMAIN_LEFT_3) $(DOMAIN_RIGHT_3) $(ITERATIONS_3)
else
	@echo "Please specify a valid set of parameters: p=1/2/3"
endif

GA:
ifeq ($(p), 1)
	$(PYTHON) algorithm/GA.py $(VECTOR_SIZE_1) $(SOLUTION_SIZE_1) $(DOMAIN_LEFT_1) $(DOMAIN_RIGHT_1) $(ITERATIONS_1)
else ifeq ($(p), 2)
	$(PYTHON) algorithm/GA.py $(VECTOR_SIZE_2) $(SOLUTION_SIZE_2) $(DOMAIN_LEFT_2) $(DOMAIN_RIGHT_2) $(ITERATIONS_2)
else ifeq ($(p), 3)
	$(PYTHON) algorithm/GA.py $(VECTOR_SIZE_3) $(SOLUTION_SIZE_3) $(DOMAIN_LEFT_3) $(DOMAIN_RIGHT_3) $(ITERATIONS_3)
else
	@echo "Please specify a valid set of parameters: p=1/2/3"
endif

show:
	$(PYTHON) algorithm/show.py