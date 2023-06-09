PYTHON?=python -X dev

all: help

help: 			## Show this help
	@echo -e "Specify a command. The choices are:\n"
	@grep -E '^[0-9a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[0;36m%-12s\033[m %s\n", $$1, $$2}'
	@echo ""
.PHONY: help

# {{{ linting

fmt: black		## Run all formatting scripts
	$(PYTHON) -m pyproject_fmt --indent 4 pyproject.toml
	$(PYTHON) -m isort pycaputo tests examples
.PHONY: fmt

black:			## Run black over the source code
	$(PYTHON) -m black \
		--safe --target-version py38 --preview \
		pycaputo tests examples docs
.PHONY: black

ruff:			## Run ruff checks over the source code
	ruff check pycaputo tests examples
	@echo -e "\e[1;32mruff clean!\e[0m"
.PHONY: ruff

mypy:			## Run mypy checks over the source code
	$(PYTHON) -m mypy \
		--strict --show-error-codes \
		pycaputo tests examples
	@echo -e "\e[1;32mmypy clean!\e[0m"
.PHONY: mypy

reuse:			## Check REUSE license compliance
	$(PYTHON) -m reuse lint
	@echo -e "\e[1;32mREUSE compliant!\e[0m"
.PHONY: reuse

codespell:		## Run codespell over the source code and documentation
	@codespell --summary \
		--skip _build \
		--ignore-words .codespell-ignore \
		pycaputo tests examples docs
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

ctags:			## Regenerate ctags
	ctags --recurse=yes \
		--tag-relative=yes \
		--exclude=.git \
		--exclude=docs \
		--python-kinds=-i \
		--language-force=python
.PHONY: ctags
