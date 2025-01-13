# Python Virtual Environments

!!! abstract "Learning Objectives"
    - Understand why virtual environments are crucial for reproducible research
    - Learn how to create and manage Python virtual environments
    - Master dependency management with requirements.txt
    - Develop best practices for environment management in scientific projects

## Why Virtual Environments?

Virtual environments solve several critical problems:

1. **Isolation**: Keep project dependencies separate
2. **Reproducibility**: Ensure consistent package versions across different machines
3. **Conflict Prevention**: Avoid package version conflicts between projects
4. **Clean Management**: Easy to create, delete, and recreate environments

![venv](./venv.assets/venv.jpg)

## Virtual Environment Tools

### venv (Built-in)
Python's built-in solution, simple and lightweight:
```bash
# Create virtual environment
python -m venv myenv

# Activate virtual environment
# On Unix/macOS:
source myenv/bin/activate
# On Windows:
myenv\Scripts\activate

# Deactivate
deactivate
```

There are other tools for python virtual environment management, such as `conda`, `pyenv`, `pipenv`, `poetry`, `uv`, etc. You can choose one of them to manage your dependencies. Here we will focus on `venv` and `requirements.txt`.

## Managing Dependencies

### pip Requirements

1. **Generate requirements.txt**:
```bash
# Manual: List all installed packages
pip freeze > requirements.txt
```

2. **Install from requirements.txt**:
```bash
pip install -r requirements.txt
```


## Recommended Workflow

1. **Project Setup**:
```bash
# Create virtual environment
python -m venv venv
```

2. **Development Workflow**:
```bash
# 1. Activate environment
source venv/bin/activate

# 2. Install/update dependencies
pip install package_name
pip freeze > requirements.txt

# 3. Code and test

# 4. Deactivate when done
deactivate
```

3. **Collaboration Workflow**:
```bash
# 1. Clone repository
git clone project_url

# 2. Create environment
make env

# 3. Activate environment
source venv/bin/activate

# 4. Work on project

# 5. Update dependencies if needed
make requirements
```

4. **Deployment Workflow**:
```bash
# 1. Create fresh environment
python -m venv prod_env

# 2. Install only production dependencies
pip install -r requirements.txt --no-dev

# 3. Run tests
pytest

# 4. Deploy
```


## Resources

- [Python venv documentation](https://docs.python.org/3/library/venv.html)
- [pip-tools documentation](https://pip-tools.readthedocs.io/)
