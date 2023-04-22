PYTHON=python
PIP=pip

# 参数组 1 
VECTOR_SIZE_1=5 # 维度较低。

# 参数组 2
VECTOR_SIZE_2=30 # 维度适中。

# 参数组 3
VECTOR_SIZE_3=50 # 维度较高。

MAKEFLAGS += --always-make # 避免使用缓存。

all: BBO-Pro BBO GA

install: requirements.txt
	$(PIP) install -r $^

depend: requirements.txt
	$(PIP) freeze > $^

BBO-Pro:
ifeq ($(p), 1)
	$(PYTHON) algorithm/BBO-Pro.py $(VECTOR_SIZE_1) 
else ifeq ($(p), 2)
	$(PYTHON) algorithm/BBO-Pro.py $(VECTOR_SIZE_2) 
else ifeq ($(p), 3)
	$(PYTHON) algorithm/BBO-Pro.py $(VECTOR_SIZE_3) 
else
	@echo "Please specify a valid set of parameters: p=1/2/3"
endif


BBO: 
ifeq ($(p), 1)
	$(PYTHON) algorithm/BBO.py $(VECTOR_SIZE_1) 
else ifeq ($(p), 2)
	$(PYTHON) algorithm/BBO.py $(VECTOR_SIZE_2) 
else ifeq ($(p), 3)
	$(PYTHON) algorithm/BBO.py $(VECTOR_SIZE_3) 
else
	@echo "Please specify a valid set of parameters: p=1/2/3"
endif

GA:
ifeq ($(p), 1)
	$(PYTHON) algorithm/GA.py $(VECTOR_SIZE_1) 
else ifeq ($(p), 2)
	$(PYTHON) algorithm/GA.py $(VECTOR_SIZE_2) 
else ifeq ($(p), 3)
	$(PYTHON) algorithm/GA.py $(VECTOR_SIZE_3) 
else
	@echo "Please specify a valid set of parameters: p=1/2/3"
endif

show:
	$(PYTHON) algorithm/show.py

all-app: BBO-Pro-app BBO-app GA-app

BBO-Pro-app:
	$(PYTHON) application/BBO-Pro.py

BBO-app:
	$(PYTHON) application/BBO.py

GA-app:
	$(PYTHON) application/GA.py

show-app:
	$(PYTHON) application/show.py
