PYTHON?=python -X dev

all: help

help: 			## Show this help
	@echo -e "Specify a command. The choices are:\n"
	@grep -E '^[0-9a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[0;36m%-12s\033[m %s\n", $$1, $$2}'
	@echo ""
.PHONY: help

# {{{ linting

format: black	## Run all formatting scripts
	$(PYTHON) -m pyproject_fmt --indent 4 pyproject.toml
	$(PYTHON) -m isort pycaputo tests examples scripts
.PHONY: format

fmt: format
.PHONY: fmt

black:			## Run black over the source code
	$(PYTHON) -m black pycaputo tests examples docs scripts
.PHONY: black

lint: ruff mypy reuse codespell manifest	## Run all linting scripts
.PHONY: lint

ruff:			## Run ruff checks over the source code
	ruff check pycaputo tests examples scripts
	@echo -e "\e[1;32mruff clean!\e[0m"
.PHONY: ruff

mypy:			## Run mypy checks over the source code
	$(PYTHON) -m mypy pycaputo tests examples scripts
	@echo -e "\e[1;32mmypy clean!\e[0m"
.PHONY: mypy

reuse:			## Check REUSE license compliance
	$(PYTHON) -m reuse lint
	@echo -e "\e[1;32mREUSE compliant!\e[0m"
.PHONY: reuse

codespell:		## Run codespell over the source code and documentation
	@codespell --summary \
		--skip _build \
		--uri-ignore-words-list '*' \
		--ignore-words .codespell-ignore \
		pycaputo tests examples docs scripts
.PHONY: codespell

manifest:		## Update MANIFEST.in file
	$(PYTHON) -m check_manifest
	@echo -e "\e[1;32mMANIFEST.in is up to date!\e[0m"
.PHONY: manifest

# }}}

# {{{ testing

REQUIREMENTS=\
	requirements-dev.txt \
	requirements.txt

requirements-dev.txt: pyproject.toml
	$(PYTHON) -m piptools compile \
		--resolver=backtracking --upgrade \
		--extra dev \
		-o $@ $<

requirements.txt: pyproject.toml
	$(PYTHON) -m piptools compile \
		--resolver=backtracking --upgrade \
		-o $@ $<

pin: $(REQUIREMENTS)	## Pin dependencies versions to requirements.txt
.PHONY: pin

pip-install:			## Install pinned depdencies from requirements.txt
	$(PYTHON) -m pip install --upgrade pip setuptools
	$(PYTHON) -m pip install -r requirements-dev.txt -e .
.PHONY: pip-install

test:					## Run pytest tests
	$(PYTHON) -m pytest -rswx --durations=25 -v -s
.PHONY: test

examples:				## Run examples
	@for ex in $$(find examples -name "*.py"); do \
		echo -e "\x1b[1;32m===> \x1b[97mRunning $${ex}\x1b[0m"; \
		$(PYTHON) "$${ex}"; \
		sleep 1; \
	done
.PHONY: examples

# }}}

# {{{ development

ctags:			## Regenerate ctags
	ctags --recurse=yes \
		--tag-relative=yes \
		--exclude=.git \
		--exclude=docs \
		--python-kinds=-i \
		--language-force=python
.PHONY: ctags

generate-doc-figures:		## Regenerate figures used in the docs.
	$(PYTHON) scripts/generate-doc-figures.py doc
	@export PYCAPUTO_SAVEFIG=svg
	@export PYCAPUTO_LOGGING_LEVEL=ERROR
	@export PYCAPUTO_DARK=no
	$(PYTHON) examples/van-der-pol-adaptive-pece.py

	export PYCAPUTO_DARK=no
	$(PYTHON) examples/van-der-pol-adaptive-pece.py

# }}}

