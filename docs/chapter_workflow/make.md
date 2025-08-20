# GNU Make 
!!! abstract "Learning Objectives"
    - Understand how Make can automate and document workflows
    - Learn Make syntax and basic concepts
    - Create Makefiles for common scientific computing tasks
    - Integrate Make with version control and data analysis pipelines

## Why Make?

Make is a powerful tool for:

1. **Automation**: Automate complex workflows with a single command
2. **Documentation**: Self-documenting build process; Makefiles are your cheat sheet
3. **Dependency Management**: Only rebuild what's necessary
4. **Reproducibility**: Ensure consistent builds across different environments

## Make Basics

### Makefile Structure

A basic Makefile consists of rules with targets, prerequisites, and recipes:

```makefile
target: prerequisites
    recipe
```

Example:
```makefile
latex: paper.tex references.bib
    pdflatex paper.tex
    bibtex paper
    pdflatex paper.tex
    pdflatex paper.tex
```

### Run Makefile commands

Put your `Makefile` in your project root directory. Then you can run the following commands to run the Makefile:
```bash
make target
```
This will run the target in the Makefile. For example, if your `Makefile` has the `latex` target as the previous example, you can run the following command to compile the LaTeX file:
```bash
make latex
```


## Common Workflows

### Version Control Workflow

```makefile
.PHONY: commit push pull clean

# Git commands with automatic messages
commit:
    git add .
    @read -p "Enter commit message: " msg; \
    git commit -m "$$msg"

push: commit
    git pull origin main
    git push origin main

pull:
    git pull origin main

# Clean generated files
clean:
    rm -f *.aux *.log *.bbl *.blg *.out
```

### LaTeX Document Workflow

```makefile
# Variables for file names
PAPER = paper
FIGS = $(wildcard figures/*.pdf)

# Main targets
.PHONY: all clean

all: $(PAPER).pdf

# Build PDF with automatic bibliography
$(PAPER).pdf: $(PAPER).tex bibliography.bib $(FIGS)
    pdflatex $(PAPER)
    bibtex $(PAPER)
    pdflatex $(PAPER)
    pdflatex $(PAPER)

# Clean LaTeX auxiliary files
clean:
    rm -f *.aux *.log *.bbl *.blg *.out $(PAPER).pdf

# Watch for changes and rebuild
watch:
    while true; do \
        make all; \
        inotifywait -e modify $(PAPER).tex bibliography.bib; \
    done
```

### Data Analysis Workflow

```makefile
# Directories
DATA_DIR = data
RESULTS_DIR = results
FIGS_DIR = figures

# Data files
RAW_DATA = $(DATA_DIR)/raw_data.csv
CLEAN_DATA = $(DATA_DIR)/clean_data.csv
ANALYSIS_RESULTS = $(RESULTS_DIR)/analysis_results.csv
FIGURES = $(FIGS_DIR)/figure1.pdf $(FIGS_DIR)/figure2.pdf

# Main targets
.PHONY: all clean

all: report.pdf

# Data processing pipeline
$(CLEAN_DATA): $(RAW_DATA) scripts/clean_data.R
    Rscript scripts/clean_data.R

$(ANALYSIS_RESULTS): $(CLEAN_DATA) scripts/analyze_data.R
    Rscript scripts/analyze_data.R

$(FIGS_DIR)/%.pdf: $(ANALYSIS_RESULTS) scripts/make_figures.R
    Rscript scripts/make_figures.R

# Generate report
report.pdf: report.Rmd $(ANALYSIS_RESULTS) $(FIGURES)
    R -e "rmarkdown::render('report.Rmd')"

# Clean generated files
clean:
    rm -f $(CLEAN_DATA) $(ANALYSIS_RESULTS) $(FIGURES) report.pdf
```

## Advanced Make Features

### Pattern Rules

Use pattern rules to handle multiple similar files:

```makefile
# Convert all .tex files to .pdf
%.pdf: %.tex
    pdflatex $<
```

### Variables and Functions

```makefile
# Variables
CC = gcc
CFLAGS = -Wall -O2

# Functions
SOURCES = $(wildcard src/*.c)
OBJECTS = $(SOURCES:.c=.o)
```

### Automatic Variables

- `$@`: Target name
- `$<`: First prerequisite
- `$^`: All prerequisites
- `$*`: Stem in pattern rule

Example:
```makefile
%.o: %.c
    $(CC) -c $(CFLAGS) $< -o $@
```

## Best Practices

### 1. Directory Structure

Organize your project with clear directory structure:
```makefile
# Directory structure
DIRS = data src results figures docs
$(shell mkdir -p $(DIRS))
```

### 2. Documentation

Include comments and help target:
```makefile
.PHONY: help
help:
    @echo "Available targets:"
    @echo "  all      - Build everything"
    @echo "  clean    - Remove generated files"
    @echo "  data     - Process raw data"
    @echo "  figures  - Generate figures"
```

### 3. Error Handling

Use error checking in recipes:
```makefile
data/processed.csv: data/raw.csv scripts/process.R
    Rscript scripts/process.R || (rm -f $@; exit 1)
```

### 4. Dependency Tracking

Track both code and data dependencies:
```makefile
results/model.rds: src/train_model.R data/training.csv
    Rscript $< data/training.csv $@
```

## Example: Complete Research Project

```makefile
# Configuration
R_SCRIPTS = $(wildcard scripts/*.R)
TEX_FILES = $(wildcard paper/*.tex)
BIB_FILES = $(wildcard paper/*.bib)

# Main targets
.PHONY: all paper data clean

all: paper data

# Paper compilation
paper: paper/manuscript.pdf

paper/manuscript.pdf: $(TEX_FILES) $(BIB_FILES) results/analysis.rds figures/*.pdf
    cd paper && pdflatex manuscript
    cd paper && bibtex manuscript
    cd paper && pdflatex manuscript
    cd paper && pdflatex manuscript

# Data analysis pipeline
data: results/analysis.rds figures/plot1.pdf figures/plot2.pdf

results/analysis.rds: scripts/analyze.R data/clean_data.csv
    Rscript $<

figures/%.pdf: scripts/plot.R results/analysis.rds
    Rscript $<

# Data cleaning
data/clean_data.csv: scripts/clean.R data/raw_data.csv
    Rscript $<

# Utility targets
clean:
    rm -f paper/*.{aux,log,bbl,blg}
    rm -f results/* figures/*

.PHONY: sync
sync: all
    git add .
    git commit -m "Update analysis and paper"
    git push origin main
```

## Resources

- [GNU Make Documentation](https://www.gnu.org/software/make/manual/)
- [Make for Data Science](https://blog.mindlessness.life/makefile/2019/11/17/the-language-agnostic-all-purpose-incredible-makefile.html)
- [Minimal Make](https://kbroman.org/minimal_make/)
