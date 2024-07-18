PYTHON?=python -X dev

all: help

help: 			## Show this help
	@echo -e "Specify a command. The choices are:\n"
	@grep -E '^[0-9a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[0;36m%-12s\033[m %s\n", $$1, $$2}'
	@echo ""
.PHONY: help

# {{{ linting

format: black isort pyproject							## Run all formatting scripts
.PHONY: format

fmt: format
.PHONY: fmt

black:			## Run ruff format over the source code
	ruff format src tests examples docs scripts
	@echo -e "\e[1;32mruff format clean!\e[0m"
.PHONY: black

isort:			## Run ruff isort fixes over the source code
	ruff check --fix --select=I src tests examples scripts docs
	ruff check --fix --select=RUF022 src
	@echo -e "\e[1;32mruff isort clean!\e[0m"
.PHONY: isort

pyproject:		## Run pyproject-fmt over the configuration
	$(PYTHON) -m pyproject_fmt --indent 4 pyproject.toml
	@echo -e "\e[1;32mpyproject clean!\e[0m"
.PHONY: pyproject

lint: ruff mypy doc8 reuse typos					## Run all linting scripts
.PHONY: lint

ruff:			## Run ruff checks over the source code
	ruff check src tests examples docs scripts
	@echo -e "\e[1;32mruff lint clean!\e[0m"
.PHONY: ruff

mypy:			## Run mypy checks over the source code
	$(PYTHON) -m mypy src tests examples scripts
	@echo -e "\e[1;32mmypy clean!\e[0m"
.PHONY: mypy

doc8:			## Run doc8 checks over the source code
	$(PYTHON) -m doc8 src docs
	@echo -e "\e[1;32mdoc8 clean!\e[0m"
.PHONY: doc8

reuse:			## Check REUSE license compliance
	$(PYTHON) -m reuse lint
	@echo -e "\e[1;32mREUSE compliant!\e[0m"
.PHONY: reuse

typos:			## Run typos over the source code and documentation
	@typos
	@echo -e "\e[1;32mtypos clean!\e[0m"
.PHONY: typos

# }}}

# {{{ testing

REQUIREMENTS=\
	requirements-dev.txt \
	requirements.txt

requirements-dev.txt: pyproject.toml
	uv pip compile --upgrade --universal --python-version '3.9' \
		--extra dev \
		-o $@ $<
.PHONY: requirements-dev.txt

requirements.txt: pyproject.toml
	uv pip compile --upgrade --universal --python-version '3.9' \
		-o $@ $<
.PHONY: requirements.txt

pin: $(REQUIREMENTS)	## Pin dependencies versions to requirements.txt
.PHONY: pin

pip-install:			## Install pinned dependencies from requirements.txt
	$(PYTHON) -m pip install --upgrade pip hatchling wheel
	$(PYTHON) -m pip install -r requirements-dev.txt -e .
.PHONY: pip-install

test:					## Run pytest tests
	$(PYTHON) -m pytest -m 'not benchmark'
.PHONY: test

benchmark:				## Run pytest benchmarks
	$(PYTHON) -m pytest -m 'benchmark' \
		--benchmark-autosave \
		--benchmark-storage=docs/benchmarks \
		--benchmark-name=short \
		--benchmark-columns=min,max,mean,iqr,ops,rounds \
		tests/benchmarks
.PHONY: benchmark

examples:				## Run examples
	@for ex in $$(find examples -name "*.py"); do \
		echo -e "\x1b[1;32m===> \x1b[97mRunning $${ex}\x1b[0m"; \
		$(PYTHON) "$${ex}"; \
		sleep 1; \
	done
.PHONY: examples

# }}}

# {{{ development

generate-doc-figures:		## Regenerate figures used in the docs.
	$(PYTHON) scripts/generate-doc-figures.py docs/_static

ctags:						## Regenerate ctags
	ctags --recurse=yes \
		--tag-relative=yes \
		--exclude=.git \
		--exclude=docs \
		--python-kinds=-i \
		--language-force=python
.PHONY: ctags

clean:						## Remove various build artifacts
	rm -rf *.png
	rm -rf build dist
	rm -rf docs/_build
.PHONY: clean

purge: clean				## Remove various temporary files
	rm -rf .ruff_cache .pytest_cache .mypy_cache
.PHONY: purge

# }}}

