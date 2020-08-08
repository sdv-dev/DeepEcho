.DEFAULT_GOAL := help

define BROWSER_PYSCRIPT
import os, webbrowser, sys

try:
	from urllib import pathname2url
except:
	from urllib.request import pathname2url

webbrowser.open("file://" + pathname2url(os.path.abspath(sys.argv[1])))
endef
export BROWSER_PYSCRIPT

define PRINT_HELP_PYSCRIPT
import re, sys

for line in sys.stdin:
	match = re.match(r'^([a-zA-Z_-]+):.*?## (.*)$$', line)
	if match:
		target, help = match.groups()
		print("%-20s %s" % (target, help))
endef
export PRINT_HELP_PYSCRIPT

BROWSER := python -c "$$BROWSER_PYSCRIPT"

.PHONY: help
help:
	@python -c "$$PRINT_HELP_PYSCRIPT" < $(MAKEFILE_LIST)


# CLEAN TARGETS

.PHONY: clean-build
clean-build: ## remove build artifacts
	rm -fr build/
	rm -fr dist/
	rm -fr .eggs/
	rm -fr benchmark/build/
	rm -fr benchmark/dist/
	rm -fr benchmark/.eggs/
	find . -name '*.egg-info' -exec rm -fr {} +
	find . -name '*.egg' -exec rm -f {} +

.PHONY: clean-pyc
clean-pyc: ## remove Python file artifacts
	find . -name '*.pyc' -exec rm -f {} +
	find . -name '*.pyo' -exec rm -f {} +
	find . -name '*~' -exec rm -f {} +
	find . -name '__pycache__' -exec rm -fr {} +

.PHONY: clean-docs
clean-docs: ## remove previously built docs
	rm -f docs/api/*.rst
	rm -rf docs/tutorials
	-$(MAKE) -C docs clean 2>/dev/null  # this fails if sphinx is not yet installed

.PHONY: clean-coverage
clean-coverage: ## remove coverage artifacts
	rm -f .coverage
	rm -f .coverage.*
	rm -fr htmlcov/

.PHONY: clean-test
clean-test: ## remove test artifacts
	rm -fr .tox/
	rm -fr .pytest_cache

.PHONY: clean
clean: clean-build clean-pyc clean-test clean-coverage clean-docs ## remove all build, test, coverage, docs and Python artifacts


# INSTALL TARGETS

.PHONY: install
install: clean-build clean-pyc ## install the package to the active Python's site-packages
	pip install .

.PHONY: install-benchmark
install-benchmark: clean-build clean-pyc ## install the package and test dependencies
	pip install ./benchmark

.PHONY: install-test
install-test: clean-build clean-pyc ## install the package and test dependencies
	pip install .[test] ./benchmark

.PHONY: install-develop
install-develop: clean-build clean-pyc ## install the package in editable mode and dependencies for development
	pip install -e .[dev] -e ./benchmark[dev]


# LINT TARGETS

.PHONY: lint-deepecho
lint-deepecho: ## check style with flake8 and isort
	flake8 deepecho
	isort -c --recursive deepecho
	pylint deepecho --rcfile=setup.cfg

.PHONY: lint-benchmark
lint-benchmark: ## check style with flake8 and isort
	flake8 benchmark/deepecho
	isort -c --recursive benchmark/deepecho
	pylint benchmark/deepecho --rcfile=setup.cfg

.PHONY: lint-tests
lint-tests: ## check style with flake8 and isort
	flake8 --ignore=D tests benchmark/tests
	isort -c --recursive tests benchmark/tests

.PHONY: lint
lint:lint-deepecho lint-benchmark lint-tests  ## Run all code style checks

.PHONY: fix-lint
fix-lint: ## fix lint issues using autoflake, autopep8, and isort
	find deepecho benchmark/deepecho tests -name '*.py' | xargs autoflake --in-place --remove-all-unused-imports --remove-unused-variables
	autopep8 --in-place --recursive --aggressive deepecho benchmark/deepecho tests
	isort --apply --atomic --recursive deepecho benchmark/deepecho tests


# TEST TARGETS

.PHONY: test-unit
test-unit: ## run tests quickly with the default Python
	python -m pytest --cov=deepecho

.PHONY: test-readme
test-readme: ## run the readme snippets
	rm -rf tests/readme_test && mkdir tests/readme_test
	cd tests/readme_test && rundoc run --single-session python3 -t python3 ../../README.md
	rm -rf tests/readme_test

.PHONY: test-tutorials
test-tutorials: ## run the tutorial notebooks
	jupyter nbconvert --execute --ExecutePreprocessor.timeout=600 tutorials/*.ipynb --stdout > /dev/null

.PHONY: check-dependencies
check-dependencies: ## test if there are any broken dependencies
	pip check

.PHONY: test
test: test-unit test-readme ## test everything that needs test dependencies

.PHONY: test-devel
test-devel: check-dependencies lint docs ## test everything that needs development dependencies

.PHONY: test-all
test-all: ## run tests on every Python version with tox
	tox -r

.PHONY: coverage
coverage: ## check code coverage quickly with the default Python
	coverage run --source deepecho -m pytest
	coverage report -m
	coverage html
	$(BROWSER) htmlcov/index.html


# DOCS TARGETS

.PHONY: docs
docs: clean-docs ## generate Sphinx HTML documentation, including API docs
	cp -r tutorials docs/tutorials
	sphinx-apidoc --separate --no-toc -M -o docs/api/ deepecho
	sphinx-apidoc --separate --no-toc -M -o docs/api/ benchmark/deepecho
	$(MAKE) -C docs html

.PHONY: view-docs
view-docs: docs ## view docs in browser
	$(BROWSER) docs/_build/html/index.html

.PHONY: serve-docs
serve-docs: view-docs ## compile the docs watching for changes
	watchmedo shell-command -W -R -D -p '*.rst;*.md' -c '$(MAKE) -C docs html' .


# RELEASE TARGETS

.PHONY: dist
dist: clean ## builds source and wheel package
	python setup.py sdist
	python setup.py bdist_wheel
	cd benchmark && python setup.py sdist
	cd benchmark && python setup.py bdist_wheel
	mv benchmark/dist/* dist && rmdir benchmark/dist
	ls -l dist

.PHONY: publish-confirm
publish-confirm:
	@echo "WARNING: This will irreversibly upload a new version to PyPI!"
	@echo -n "Please type 'confirm' to proceed: " \
		&& read answer \
		&& [ "$${answer}" = "confirm" ]

.PHONY: publish-test
publish-test: dist publish-confirm ## package and upload a release on TestPyPI
	twine upload --repository-url https://test.pypi.org/legacy/ dist/*

.PHONY: publish
publish: dist publish-confirm ## package and upload a release
	twine upload dist/*

.PHONY: bumpversion-release
bumpversion-release: ## Merge master to stable and bumpversion release
	git checkout stable || git checkout -b stable
	git merge --no-ff master -m"make release-tag: Merge branch 'master' into stable"
	bumpversion release
	git push --tags origin stable

.PHONY: bumpversion-release-test
bumpversion-release-test: ## Merge master to stable and bumpversion release
	git checkout stable || git checkout -b stable
	git merge --no-ff master -m"make release-tag: Merge branch 'master' into stable"
	bumpversion release --no-tag
	@echo git push --tags origin stable

.PHONY: bumpversion-patch
bumpversion-patch: ## Merge stable to master and bumpversion patch
	git checkout master
	git merge stable
	bumpversion --no-tag patch
	git push

.PHONY: bumpversion-candidate
bumpversion-candidate: ## Bump the version to the next candidate
	bumpversion candidate --no-tag

.PHONY: bumpversion-minor
bumpversion-minor: ## Bump the version the next minor skipping the release
	bumpversion --no-tag minor

.PHONY: bumpversion-major
bumpversion-major: ## Bump the version the next major skipping the release
	bumpversion --no-tag major

.PHONY: bumpversion-revert
bumpversion-revert: ## Undo a previous bumpversion-release
	git checkout master
	git branch -D stable

CLEAN_DIR := $(shell git status --short | grep -v ??)
CURRENT_BRANCH := $(shell git rev-parse --abbrev-ref HEAD 2>/dev/null)
CHANGELOG_LINES := $(shell git diff HEAD..origin/stable HISTORY.md 2>&1 | wc -l)

.PHONY: check-clean
check-clean: ## Check if the directory has uncommitted changes
ifneq ($(CLEAN_DIR),)
	$(error There are uncommitted changes)
endif

.PHONY: check-master
check-master: ## Check if we are in master branch
ifneq ($(CURRENT_BRANCH),master)
	$(error Please make the release from master branch\n)
endif

.PHONY: check-history
check-history: ## Check if HISTORY.md has been modified
ifeq ($(CHANGELOG_LINES),0)
	$(error Please insert the release notes in HISTORY.md before releasing)
endif

.PHONY: check-release
check-release: check-clean check-master check-history ## Check if the release can be made
	@echo "A new release can be made"

.PHONY: release
release: check-release bumpversion-release publish bumpversion-patch

.PHONY: release-test
release-test: check-release bumpversion-release-test publish-test bumpversion-revert

.PHONY: release-candidate
release-candidate: check-master publish bumpversion-candidate

.PHONY: release-candidate-test
release-candidate-test: check-clean check-master publish-test

.PHONY: release-minor
release-minor: check-release bumpversion-minor release

.PHONY: release-major
release-major: check-release bumpversion-major release


# DOCKER TARGETS

.PHONY: docker-login
docker-login:
	docker login docker.io

.PHONY: docker-build
docker-build:
	docker build -t deepecho .

.PHONY: docker-push
docker-push: docker-login
	@$(eval VERSION := $(shell python -c 'import deepecho; print(deepecho.__version__)'))
	docker tag deepecho sdvproject/deepecho:$(VERSION)
	docker push sdvproject/deepecho:$(VERSION)

.PHONY: docker-publish
docker-publish: docker-login docker-build docker-push
