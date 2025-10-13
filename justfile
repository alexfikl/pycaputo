PYTHON := "python -X dev"

_default:
    @just --list

# {{{ formatting

alias fmt: format

[doc("Reformat all source code")]
format: isort black pyproject justfmt

[doc("Run ruff isort fixes over the source code")]
isort:
    ruff check --fix --select=I src tests examples docs scripts
    ruff check --fix --select=RUF022 src
    @echo -e "\e[1;32mruff isort clean!\e[0m"

[doc("Run ruff format over the source code")]
black:
    ruff format src tests examples docs scripts
    @echo -e "\e[1;32mruff format clean!\e[0m"

[doc("Run pyproject-fmt over the configuration")]
pyproject:
    {{ PYTHON }} -m pyproject_fmt \
        --indent 4 --max-supported-python "3.14" \
        pyproject.toml
    @echo -e "\e[1;32mpyproject clean!\e[0m"

[doc("Run just --fmt over the justfiles")]
justfmt:
    just --unstable --fmt
    just -f docs/justfile --unstable --fmt
    @echo -e "\e[1;32mjust --fmt clean!\e[0m"

# }}}
# {{{ linting

[doc("Run all linting checks over the source code")]
lint: typos reuse ruff doc8 mypy

[doc("Run typos over the source code and documentation")]
typos:
    typos --sort
    @echo -e "\e[1;32mtypos clean!\e[0m"

[doc("Check REUSE license compliance")]
reuse:
    {{ PYTHON }} -m reuse lint
    @echo -e "\e[1;32mREUSE compliant!\e[0m"

[doc("Run ruff checks over the source code")]
ruff:
    ruff check src tests examples docs scripts
    @echo -e "\e[1;32mruff clean!\e[0m"

[doc("Run doc8 checks over the documentation")]
doc8:
    {{ PYTHON }} -m doc8 src docs
    @echo -e "\e[1;32mdoc8 clean!\e[0m"

[doc("Run mypy checks over the source code")]
mypy:
    {{ PYTHON }} -m mypy src tests examples scripts
    @echo -e "\e[1;32mmypy clean!\e[0m"

# }}}
# {{{ pin

[private]
requirements_build_txt:
    uv pip compile --upgrade --universal --python-version "3.10" \
        -o .ci/requirements-build.txt .ci/requirements-build.in

[private]
requirements_test_txt:
    uv pip compile --upgrade --universal --python-version "3.10" \
        --extra test \
        -o .ci/requirements-test.txt pyproject.toml .ci/requirements-git.txt

[private]
requirements_txt:
    uv pip compile --upgrade --universal --python-version "3.10" \
        -o requirements.txt pyproject.toml

[doc("Pin dependency versions to requirements.txt")]
pin: requirements_txt requirements_test_txt requirements_build_txt

# }}}
# {{{ develop

[doc("Install project in editable mode")]
develop:
    @rm -rf build
    @rm -rf dist
    {{ PYTHON }} -m pip install \
        --verbose \
        --no-build-isolation \
        --editable .

[doc("Editable install using pinned dependencies from requirements-test.txt")]
pip-install:
    {{ PYTHON }} -m pip install --verbose --requirement .ci/requirements-build.txt
    {{ PYTHON }} -m pip install \
        --verbose \
        --requirement .ci/requirements-test.txt \
        --no-build-isolation \
        --editable .

[doc("Remove various build artifacts")]
clean:
    rm -rf *.png
    rm -rf build dist
    rm -rf docs/build.sphinx

[doc("Remove various temporary files and caches")]
purge: clean
    rm -rf .ruff_cache .pytest_cache .pytest-cache .mypy_cache tags

[doc("Regenerate ctags")]
ctags:
    ctags --recurse=yes \
        --tag-relative=yes \
        --exclude=.git \
        --exclude=docs \
        --python-kinds=-i \
        --language-force=python

[doc("Regenerate all figures in the documentation (see scripts/generate-doc-figures.py)")]
generate-doc-figures artifact="all":
    {{ PYTHON }} scripts/generate-doc-figures.py -s {{ artifact }} docs/_static

# }}}
# {{{ tests

[doc("Run pytest tests")]
test *PYTEST_ADDOPTS:
    {{ PYTHON }} -m pytest \
        --junit-xml=pytest-results.xml \
        {{ PYTEST_ADDOPTS }}

[doc("Run pytest benchmarks")]
benchmark:
    {{ PYTHON }} -m pytest -m "benchmark" \
        --benchmark-autosave \
        --benchmark-storage=docs/benchmarks \
        --benchmark-name=short \
        --benchmark-columns=min,max,mean,iqr,ops,rounds \
        tests/benchmarks

[doc("Run examples with default options")]
examples:
    @for ex in `ls examples/*.py`; do \
        echo "::group::Running ${ex}"; \
        {{ PYTHON }} ${ex}; \
        echo "::endgroup::"; \
    done

# }}}
