# TODO Test this Makefile
.PHONY: setup

setup:
	conda env create -f environment.yml
	conda activate relu
	pip-compile --output-file requirements.txt requirements.in
	pip install -r requirements.txt

.PHONY: clean

clean:
	conda env remove -n relu

.PHONY: help

help:
	@echo "Available targets:"
	@echo "  setup      : Create and activate the 'relu' conda environment"
	@echo "  clean      : Remove the 'relu' conda environment"
	@echo "  help       : Show this help message"
