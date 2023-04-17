PYTHON=python
PIP=pip

MAKEFLAGS += --always-make # 避免使用缓存。

all: BBO-Pro BBO GA

install: requirements.txt
	$(PIP) install -r $^

depend: requirements.txt
	$(PIP) freeze > $^

BBO-Pro:
	$(PYTHON) algorithm/BBO-Pro.py

BBO: 
	$(PYTHON) algorithm/BBO.py

GA:
	$(PYTHON) algorithm/GA.py

show:
	$(PYTHON) algorithm/show.py