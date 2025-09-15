# -----------------
# Local Development
# -----------------
# -------- Local dev with a venv ----------

PY ?= python3
VENV ?= .venv
PIP := $(VENV)/bin/pip
PYTHON := $(VENV)/bin/python

venv:
	$(PY) -m venv $(VENV)

install: venv
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt

format:
	$(VENV)/bin/black *.py

lint:
	$(VENV)/bin/flake8 *.py

test:
	$(VENV)/bin/pytest -vv

run-local:
	$(PYTHON) analysis.py

clean:
	rm -rf __pycache__ .pytest_cache .coverage outputs/*

all: install format lint test

# -----------------
# Docker Workflow
# -----------------
IMAGE_NAME=ids706-wk3
CONTAINER_NAME=ids706-wk3

docker-build:
	docker build -t $(IMAGE_NAME) .

docker-run:
	docker run -it --name $(CONTAINER_NAME) \
		-v $(PWD)/outputs:/app/outputs \
		$(IMAGE_NAME):latest

image_show:
	docker images

container_show:
	docker ps -a

docker_clean:
	docker rm -f $(CONTAINER_NAME) || true
	docker rmi -f $(IMAGE_NAME) || true