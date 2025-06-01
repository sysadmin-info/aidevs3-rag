.PHONY: help install test run clean

help:
	@echo "Available commands:"
	@echo "  make install    - Install dependencies"
	@echo "  make test       - Run tests"
	@echo "  make run        - Run the main application"
	@echo "  make clean      - Clean cache and temporary files"

install:
	pip install -r requirements.txt

test:
	pytest tests/ -v

run:
	python main.py

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	rm -rf data/cache/*
	rm -rf logs/*
