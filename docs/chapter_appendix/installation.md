# Installation Guide for the Course

## Operating System

Most of the materials in this course are platform-independent. The instructions below include steps for all major operating systems: Linux, macOS, and Windows.

## Install IDE

For your local Integrated Development Environment (IDE), we suggest using Visual Studio Code (VS Code) - a lightweight, open-source editor. Head to the [VS Code official website](https://code.visualstudio.com/) to download and install the version that matches your operating system.

## Create a GitHub account

### Github Copilot

We recommend using the AI assistant Github Copilot or other similar tools. As Harvard students, you may apply for a free license of GitHub Copilot by following the steps [here](https://docs.github.com/en/education/explore-the-benefits-of-teaching-and-learning-with-github-education/github-education-for-students/apply-to-github-education-as-a-student). Note that you will need to upload proof of enrollment: you can use a picture of your student ID your GSAS enrollment documentation (which can be found in my.harvard). 

To install Github Copilot in VS Code, you can search for `Github Copilot` in the VS Code extension marketplace and install it.

We also strongly recommend you to read the VS Code [tutorial](https://code.visualstudio.com/docs/copilot/overview) for Github Copilot. It introduces many useful features of Github Copilot.

### Other AI-Copilot resources

We also recommend (though not required) to install the following AI-Copilot tools:
- [Cursor](https://www.cursor.sh/)
- [Windsurf](https://codeium.com/windsurf)

## Install language environments

### Python

We suggest using pyenv to install python for better version management. The installation process varies by operating system:

#### For macOS:

1. Install pyenv:
    ```bash
    brew install pyenv
    ```

2. Add pyenv to your shell configuration:
    - For bash, add to `~/.bashrc`:
    - For zsh, add to `~/.zshrc`:
    ```bash
    export PYENV_ROOT="$HOME/.pyenv"
    export PATH="$PYENV_ROOT/bin:$PATH"
    eval "$(pyenv init -)"
    ```

#### For Linux:

1. Install pyenv:
    ```bash
    curl https://pyenv.run | bash
    ```

2. Add the same configuration as macOS to your `~/.bashrc` or `~/.zshrc`

#### For Windows:

1. Install pyenv-win using PowerShell (run as Administrator):
    ```powershell
    Invoke-WebRequest -UseBasicParsing -Uri "https://raw.githubusercontent.com/pyenv-win/pyenv-win/master/pyenv-win/install-pyenv-win.ps1" -OutFile "./install-pyenv-win.ps1"; &"./install-pyenv-win.ps1"
    ```

2. Add System Environment Variables:
    - Open System Properties > Advanced > Environment Variables
    - Add to System Variables:
        - PYENV: `%USERPROFILE%\.pyenv\pyenv-win`
        - PYENV_HOME: `%USERPROFILE%\.pyenv\pyenv-win`
    - Add to Path:
        - `%USERPROFILE%\.pyenv\pyenv-win\bin`
        - `%USERPROFILE%\.pyenv\pyenv-win\shims`

For all operating systems, after installation:

1. Install Python:
   ```bash
   pyenv install 3.10.0
   pyenv global 3.10.0
   ```

2. Verify the installation:
   ```bash
   python --version  # Should show Python 3.10.0
   ```

#### Check pip installation

Pip is Python's package installer. It usually comes with Python, but it's good to verify the installation:

1. Check if pip is installed:
```bash
pip --version
```

If pip is not installed or you need to upgrade it:

#### For macOS/Linux:
```bash
curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py
python get-pip.py
```

#### For Windows:
```powershell
python -m ensurepip --upgrade
```

After installation, verify pip is working:
```bash
pip --version
```

#### Jupyter Notebook

The installation process is the same for all operating systems:

1. Install Jupyter using pip:
   ```bash
   pip install notebook
   ```

2. Verify the installation:
   ```bash 
   jupyter notebook --version
   ```

3. Launch Jupyter Notebook:
   ```bash
   jupyter notebook
   ```

#### VS Code Python

Install the following extensions from VS Code marketplace (same for all operating systems):
- Python
- Jupyter
- Pylance
- Pylint

### R 

#### For macOS:
- Download and install R from [CRAN](https://cran.r-project.org/bin/macosx/)

#### For Linux:
```bash
sudo apt-get update
sudo apt-get install r-base
```

#### For Windows:
- Download and install R from [CRAN](https://cran.r-project.org/bin/windows/base/)
- Download and install Rtools from [CRAN](https://cran.r-project.org/bin/windows/Rtools/)

For all operating systems:

1. Install radian:
```bash
pip install -U radian
```

2. Install required R packages from R console:
```r
install.packages("languageserver")
install.packages("httpgd")
```

3. Configure VS Code:
- Install the R extension
- Set radian path in VS Code settings:
  - For Windows: Set `r.rterm.windows` to the path of radian (typically `%USERPROFILE%\AppData\Local\Programs\Python\Python3x\Scripts\radian.exe`)
  - For macOS/Linux: Set `r.rterm.mac` or `r.rterm.linux` to the output of `which radian`

## Optional software

### Cursor

Cursor is available for all operating systems. Download from the [official website](https://www.cursor.sh/).

### Conda

#### For macOS:
- Download the appropriate installer (Apple Silicon or Intel) from [Miniconda download page](https://docs.conda.io/en/latest/miniconda.html)
- Install using the .pkg installer

#### For Linux:
```bash
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh
```

#### For Windows:
- Download the Windows installer from [Miniconda download page](https://docs.conda.io/en/latest/miniconda.html)
- Run the .exe installer
- During installation, check "Add Miniconda3 to my PATH environment variable"

For all operating systems, after installation:

1. Initialize conda:
```bash
conda init
```

2. Verify installation:
```bash
conda --version
```

3. Run `conda deactivate` to leave the environment.

For more detailed information, refer to the [official Miniconda installation documentation](https://docs.anaconda.com/miniconda/install/).





