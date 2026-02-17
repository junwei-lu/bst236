# R Version Control

!!! abstract "Learning Objectives"
    - Learn to use `here` for consistent file paths
    - Learn to use `renv` for reproducible environments
    - Develop best practices for R project management

## Why Package Management?

Package management in R is crucial for:

1. **Reproducibility**: Ensure same package versions across different machines
2. **Dependency Tracking**: Keep track of all required packages
3. **Project Isolation**: Avoid conflicts between different projects
4. **Collaboration**: Make it easy for others to recreate your environment

## No Absolute Paths with `here`

The `here` package solves the problem of absolute paths by providing a consistent way to reference files relative to your project root.

### Setting Up `here`

1. Install the package:
```r
install.packages("here")
```

2. In each R script or Rmd file, start with:
```r
here::i_am('path/to/this/file')
```

### Using `here`

Instead of absolute paths or relative paths that might break, use `here::here()`:

```r
# Bad practice
data <- read.csv("/Users/me/Projects/my_project/data/my_data.csv")

# Good practice
here::i_am('R/analysis.R')
data <- read.csv(here::here('data', 'my_data.csv'))
```

### Example Script Structure

```r
# In R/01_fit_models.R
here::i_am('R/01_fit_models.R')

# Load data
data <- read.csv(here::here('data', 'cleaned_data.csv'))

# Save results
save(results, file = here::here('output', 'model_results.RData'))
```

## Version Control with `renv`

The `renv` package manages project-specific package dependencies, ensuring reproducibility across different environments.

### Initial Setup

1. Install renv:
```r
install.packages("renv")
```

2. Initialize in your project:
```r
renv::init()
```

This creates:

- `renv.lock`: Records package versions
- `.Rprofile`: Activates renv
- `renv/`: Contains project library

### Key `renv` Commands

```r
# Install packages recorded in renv.lock
renv::restore()

# Update renv.lock with current packages
renv::snapshot()

# Remove a package
renv::remove("package_name")

# Check status
renv::status()
```

### Collaborative Workflow with `renv`

1. **Initial Setup (User A)**:
```r
renv::init()
# Commit renv.lock, .Rprofile, and renv/activate.R
```

2. **New Collaborator (User B)**:
```r
# After cloning repository
renv::restore()
```

3. **Adding New Packages (Any User)**:
```r
install.packages("new_package")
renv::snapshot()
# Commit updated renv.lock
```

4. **Syncing Changes (All Users)**:
```r
# After pulling updates
renv::restore()
```

## Recommended Workflow

1. **Project Setup**:
```r
# Create new RStudio project
# Initialize renv and here
install.packages(c("renv", "here"))
renv::init()
library(here)

# Document dependencies in DESCRIPTION file
usethis::use_description()

# Set up version control
usethis::use_git()

# Create standard directory structure
dir.create(here("data", "raw"), recursive = TRUE)
dir.create(here("data", "processed"), recursive = TRUE)
dir.create(here("R"))
dir.create(here("results", "figures"), recursive = TRUE)
```

2. **Development Workflow**:
```r
# Start new analysis script
library(here)
library(tidyverse)

# Use here for all file paths
data <- read_csv(here("data", "raw", "input.csv"))

# Install new package if needed
renv::install("newpackage")

# Work on your analysis
processed_data <- clean_data(data)
write_csv(processed_data, here("data", "processed", "clean.csv"))

# Save results using here
ggsave(here("results", "figures", "analysis_plot.pdf"))

# Update lock file
renv::snapshot()

# Document dependencies in DESCRIPTION
usethis::use_package("newpackage")
```

3. **Collaboration Workflow**:
```r
# Clone project
# Open project in RStudio (important for here to work!)

# Restore environment
renv::restore()

# Load packages
library(here)

# Work on project using relative paths
source(here("R", "analysis.R"))

# Update dependencies
renv::snapshot()
```

4. **Deployment/Sharing**:
```r
# Ensure all paths use here()
# Check no absolute paths remain:
grep -r "setwd\\|read\\.csv(" R/
grep -r "^[A-Za-z]:" R/  # Windows paths
grep -r "^/" R/          # Unix paths

# Ensure all dependencies are documented
renv::snapshot()

# Clean up unused packages
renv::clean()

# Bundle project for sharing
renv::bundle()
```



## Advanced Tips

### 1. Using Multiple R Versions

Use `.Rversion` file:
```plaintext
4.1.2
```

### 2. Custom Package Sources

In `renv.lock`:
```json
{
  "R": {
    "Version": "4.1.2",
    "Repositories": [
      {
        "Name": "CRAN",
        "URL": "https://cloud.r-project.org"
      },
      {
        "Name": "Custom",
        "URL": "https://mycran.org"
      }
    ]
  }
}
```

### 3. Continuous Integration

Example GitHub Actions workflow:
```yaml
on: [push, pull_request]

jobs:
  build:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - uses: r-lib/actions/setup-r@v2
      - uses: r-lib/actions/setup-renv@v2
      - run: renv::restore()
      - run: Rscript -e "source('analysis/analysis.R')"
```

## Resources

- [renv Documentation](https://rstudio.github.io/renv/)
- [R Packages Book](https://r-pkgs.org/)
- [usethis Package](https://usethis.r-lib.org/)
- [Writing R Extensions](https://cran.r-project.org/doc/manuals/r-release/R-exts.html)

