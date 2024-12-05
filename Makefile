# Github variables
REPO_NAME = Python-Project-Template
GITHUB_USER = junwei-lu
BRANCH = main

# Specify the desired Python version
PYTHON_VERSION = 3.12.7

# Virtual environment settings
VENV_METHOD = uv  # Change this to 'poetry', 'conda', or 'uv' as needed
VENV_NAME = venv
PYTHON = python
VENV_PIP := $(VENV_NAME)/bin/pip

CONDA_BASE := /opt/anaconda3
CONDA_ACTIVATE := source $(CONDA_BASE)/etc/profile.d/conda.sh

# Combine all PHONY targets
.PHONY: venv install update freeze activate deactivate list-packages clean dev-install format lint test help init_config init_project_structure clean-data



# Initialize a local Git repository and push to GitHub
init:
	git init
	git add .
	git commit -m "Initial commit"
	make init_repo

init_repo:
	gh repo create $(GITHUB_USER)/$(REPO_NAME) --private --source=. --remote=origin
	git push -u origin main

# Sync with Github
sync:
	pyenv local $(PYTHON_VERSION)
	@echo "Syncing with GitHub and updating packages..."
	git checkout main
	git pull origin main
	if [ "$(strip $(BRANCH))" != "main" ]; then \
		git checkout $(BRANCH); \
		git rebase main; \
	fi
	@if [ "$(strip $(VENV_METHOD))" = "venv" ]; then \
		$(VENV_PIP) install --upgrade -r requirements.txt; \
	elif [ "$(strip $(VENV_METHOD))" = "poetry" ]; then \
		poetry install; \
	elif [ "$(strip $(VENV_METHOD))" = "conda" ]; then \
		$(VENV_PIP) install --upgrade -r requirements.txt; \
	elif [ "$(strip $(VENV_METHOD))" = "uv" ]; then \
		uv pip sync requirements.txt; \
	else \
		echo "Unknown VENV_METHOD: '$(VENV_METHOD)'"; \
	fi
	@echo "Packages updated using $(VENV_METHOD)!"

# Push to GitHub
push:
	@echo "Freezing current packages to lockfile..."
	@if [ "$(strip $(VENV_METHOD))" = "venv" ]; then \
		$(VENV_PIP) freeze > requirements.txt; \
	elif [ "$(strip $(VENV_METHOD))" = "poetry" ]; then \
		poetry lock; \
	elif [ "$(strip $(VENV_METHOD))" = "conda" ]; then \
		$(CONDA_ACTIVATE) && \
		conda env export > environment.yml; \
	elif [ "$(strip $(VENV_METHOD))" = "uv" ]; then \
		uv pip freeze > requirements.txt; \
	else \
		echo "Unknown VENV_METHOD: '$(VENV_METHOD)'"; \
	fi
	echo "Lockfile created using $(VENV_METHOD)!"
	echo "Pushing to GitHub..."
	git add .
	git commit --amend --no-edit
	@if [ "$(strip $(BRANCH))" = "main" ]; then \
		git pull origin main --rebase; \
		git push origin main; \
	else \
		git push -f origin $(BRANCH); \
	fi

# Create and initialize virtual environment
venv:
	@echo "Creating virtual environment using $(VENV_METHOD) with $(PYTHON_VERSION)..."
	pyenv local $(PYTHON_VERSION)
	@if [ "$(strip $(VENV_METHOD))" = "venv" ]; then \
		$(PYTHON) -m venv $(VENV_NAME); \
		$(VENV_NAME)/bin/pip install --upgrade pip; \
	elif [ "$(strip $(VENV_METHOD))" = "poetry" ]; then \
		poetry env use $(PYTHON_VERSION); \
		poetry config virtualenvs.in-project true; \
		poetry init --no-interaction --name "project" --description "" --author "Junwei Lu" --license "MIT"; \
		sed -i '' 's/python = ".*"/python = "$(PYTHON_VERSION)"/' pyproject.toml; \
		sed -i '' 's/python = ".*"/python = "$(PYTHON_VERSION)"/' pyproject.toml; \
		poetry install; \
	elif [ "$(strip $(VENV_METHOD))" = "conda" ]; then \
		$(CONDA_ACTIVATE) && \
		conda create --prefix $(VENV_NAME) python=$(PYTHON_VERSION) -y; \
	elif [ "$(strip $(VENV_METHOD))" = "uv" ]; then \
		uv venv $(VENV_NAME); \
	else \
		echo "Unknown VENV_METHOD: '$(VENV_METHOD)'"; \
	fi
	@echo "Virtual environment created using $(VENV_METHOD) with $(PYTHON_VERSION)!"

# Install data science packages
install:
	pyenv local $(PYTHON_VERSION)
	@echo "Installing packages for data science..."
	@if [ "$(strip $(VENV_METHOD))" = "venv" ]; then \
		source $(VENV_NAME)/bin/activate; \
		$(VENV_PIP) install numpy pandas scikit-learn matplotlib seaborn jupyter; \
	elif [ "$(strip $(VENV_METHOD))" = "poetry" ]; then \
		poetry add numpy pandas scikit-learn matplotlib seaborn jupyter; \
	elif [ "$(strip $(VENV_METHOD))" = "conda" ]; then \
		$(CONDA_ACTIVATE) && \
		conda install --prefix $(VENV_NAME) numpy pandas scikit-learn matplotlib seaborn jupyter -y; \
	elif [ "$(strip $(VENV_METHOD))" = "uv" ]; then \
		source $(VENV_NAME)/bin/activate; \
		uv pip install numpy pandas scikit-learn matplotlib seaborn jupyter; \
	fi
	@echo "Packages installed using $(VENV_METHOD)!"

# Update existing packages
update_packages:
	@echo "Updating packages..."
	@if [ "$(strip $(VENV_METHOD))" = "venv" ]; then \
		$(VENV_PIP) install --upgrade -r requirements.txt; \
	elif [ "$(strip $(VENV_METHOD))" = "poetry" ]; then \
		poetry update; \
	elif [ "$(strip $(VENV_METHOD))" = "conda" ]; then \
		$(VENV_PIP) install --upgrade -r requirements.txt; \
	elif [ "$(strip $(VENV_METHOD))" = "uv" ]; then \
		uv pip sync requirements.txt; \
	fi
	@echo "Packages updated using $(VENV_METHOD)!"

# Save current package versions
lock:
	@echo "Freezing current packages to lockfile..."
	@if [ "$(strip $(VENV_METHOD))" = "venv" ]; then \
		$(VENV_PIP) freeze > requirements.txt; \
	elif [ "$(strip $(VENV_METHOD))" = "poetry" ]; then \
		poetry lock; \
	elif [ "$(strip $(VENV_METHOD))" = "conda" ]; then \
		$(CONDA_ACTIVATE) && \
		conda env export > environment.yml; \
	elif [ "$(strip $(VENV_METHOD))" = "uv" ]; then \
		uv pip freeze > requirements.txt; \
	else \
		echo "Unknown VENV_METHOD: '$(VENV_METHOD)'"; \
	fi
	@echo "Lockfile created using $(VENV_METHOD)!"


# Show activation command
activate:
	@echo "To activate the virtual environment, run:"
	@if [ "$(strip $(VENV_METHOD))" = "venv" ]; then \
		echo "source $(VENV_NAME)/bin/activate"; \
	elif [ "$(strip $(VENV_METHOD))" = "poetry" ]; then \
		poetry shell; \
	elif [ "$(strip $(VENV_METHOD))" = "conda" ]; then \
		echo "conda activate $(shell pwd)/$(VENV_NAME)"; \
	elif [ "$(strip $(VENV_METHOD))" = "uv" ]; then \
		echo "source $(VENV_NAME)/bin/activate"; \
	fi

# Show deactivation command
deactivate:
	@echo "To deactivate the virtual environment, run:"
	@if [ "$(strip $(VENV_METHOD))" = "venv" ]; then \
		deactivate; \
	elif [ "$(strip $(VENV_METHOD))" = "poetry" ]; then \
		exit; \
	elif [ "$(strip $(VENV_METHOD))" = "conda" ]; then \
		echo "conda deactivate"; \
	elif [ "$(strip $(VENV_METHOD))" = "uv" ]; then \
		deactivate; \
	else \
		echo "Unknown VENV_METHOD: '$(VENV_METHOD)'"; \
	fi

# Display installed packages
list:
	@echo "Listing installed packages for $(VENV_METHOD)..."
	@if [ "$(strip $(VENV_METHOD))" = "venv" ]; then \
		$(VENV_PIP) list; \
	elif [ "$(strip $(VENV_METHOD))" = "poetry" ]; then \
		poetry show; \
	elif [ "$(strip $(VENV_METHOD))" = "conda" ]; then \
		$(CONDA_ACTIVATE) && conda list --prefix $(VENV_NAME); \
	elif [ "$(strip $(VENV_METHOD))" = "uv" ]; then \
		uv pip list; \
	else \
		echo "Unknown VENV_METHOD: '$(VENV_METHOD)'"; \
	fi

# Remove virtual environment and cache
clean:
	@echo "VENV_METHOD: '$(VENV_METHOD)'"
	@if [ "$(strip $(VENV_METHOD))" = "conda" ]; then \
		echo "Activating conda..."; \
		$(CONDA_ACTIVATE) && echo "Removing conda environment..."; \
		conda remove --prefix $(VENV_NAME) --all -y; \
		rm -rf environment.yml; \
	elif [ "$(strip $(VENV_METHOD))" = "poetry" ]; then \
		echo "Removing poetry environment..."; \
		rm -rf .venv; \
		rm -rf *.toml; \
		rm -rf poetry.lock; \
	elif [ "$(strip $(VENV_METHOD))" = "venv" ]; then \
		echo "Removing venv environment..."; \
		rm -rf $(VENV_NAME); \
		rm -rf requirements.txt; \
	elif [ "$(strip $(VENV_METHOD))" = "uv" ]; then \
		echo "Removing uv environment..."; \
		rm -rf $(VENV_NAME); \
		rm -rf requirements.txt; \
	else \
		echo "Unknown VENV_METHOD: '$(VENV_METHOD)'"; \
	fi
	@echo "Clean complete"

branch:
	git checkout -b $(BRANCH)

push_branch:
	git push origin $(BRANCH)

small_update:
	git add .
	git commit --amend --no-edit

change_name:
	git commit --amend -m "Update"

