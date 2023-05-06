PYTHON=python
PIP=pip

# alg-参数组 1 
VECTOR_SIZE_1=5 # 维度较低。

# alg-参数组 2
VECTOR_SIZE_2=30 # 维度适中。

# alg-参数组 3
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

# app-参数组 1 
MOCK_1=mock-mid # 维度适中。

# app-参数组 2
MOCK_2=mock-high # 维度较高。

all-app: BBO-Pro-app BBO-app GA-app

BBO-Pro-app:
ifeq ($(m), 1)
	$(PYTHON) application/BBO-Pro.py $(MOCK_1) 
else ifeq ($(m), 2)
	$(PYTHON) application/BBO-Pro.py $(MOCK_2) 
else
	@echo "Please specify a valid set of parameters: m=1/2"
endif

BBO-app:
ifeq ($(m), 1)
	$(PYTHON) application/BBO.py $(MOCK_1) 
else ifeq ($(m), 2)
	$(PYTHON) application/BBO.py $(MOCK_2) 
else
	@echo "Please specify a valid set of parameters: m=1/2"
endif

GA-app:
ifeq ($(m), 1)
	$(PYTHON) application/GA.py $(MOCK_1) 
else ifeq ($(m), 2)
	$(PYTHON) application/GA.py $(MOCK_2) 
else
	@echo "Please specify a valid set of parameters: m=1/2"
endif

show-app:
	$(PYTHON) application/show.py
