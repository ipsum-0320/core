PYTHON=python
PIP=pip
VECTOR_SIZE=6
SOLUTION_SIZE=25
DOMAIN_LEFT=-10
DOMAIN_RIGHT=10
ITERATIONS=50

MAKEFLAGS += --always-make # 避免使用缓存。

all: BBO-Pro BBO GA

install: requirements.txt
	$(PIP) install -r $^

depend: requirements.txt
	$(PIP) freeze > $^

BBO-Pro:
	$(PYTHON) algorithm/BBO-Pro.py $(VECTOR_SIZE) $(SOLUTION_SIZE) $(DOMAIN_LEFT) $(DOMAIN_RIGHT) $(ITERATIONS)

BBO: 
	$(PYTHON) algorithm/BBO.py $(VECTOR_SIZE) $(SOLUTION_SIZE) $(DOMAIN_LEFT) $(DOMAIN_RIGHT) $(ITERATIONS)

GA:
	$(PYTHON) algorithm/GA.py $(VECTOR_SIZE) $(SOLUTION_SIZE) $(DOMAIN_LEFT) $(DOMAIN_RIGHT) $(ITERATIONS)

show:
	$(PYTHON) algorithm/show.py