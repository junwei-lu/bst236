# Scientific Computing Workflow

The materials here are based on the paper [Good Enough Practices for Scientific Computing](https://swcarpentry.github.io/good-enough-practices-in-scientific-computing/).
We will simplify the paper to summarize the step by step workflow of a scientific computing project. This is only a suggested workflow, you can customize it based on your needs.

## Basic principles

* Put everything in one version-controlled directory.
* Be consistent, but look for ways to improve:
      * naming conventions, file structure, `make` structure
* Raw data are sacred. Keep them separate from everything else. 
* Separate code and data.
* Use `make` files and/or READMEs to document dependencies.
* No spaces in file names. 
* Use meaningful file names.
* Use YYYY-MM-DD date formatting.
* No absolute paths.
* Use a package management system.

## Tools 

Here are the tools you could use to implement the principles above:

- **Version Control**: Git 
- **Workflow Automation**: Make and Docker
- **Dependency Management**: Python virtual environment, R `renv` package


## Project Management

Different projects have different needs:
- Data analysis
- First author paper
- Talks, etc.

You should think about the organization of each type of project from outset. Create a project template and reuse it for each project with the help of cloud storage and bash. Here is an example:
```bash
#!/bin/bash
# Write an alias for git clone from your project template
echo "alias project_type='git clone https://github.com/your_username/your_project_template.git'" >> ~/.bashrc && 
# Apply the changes to the current session
source ~/.bashrc 
```
To make your workflow transferable across different machines, you can use cloud storage to sync your dotfiles as well.

### Project Organization Guidelines

Here is an example of a project structure:
```
project/
├── refs/
├── README.md
├── Makefile
├── data/
│   └── birds_count_table.csv
├── doc/
│   ├── notebook.md
│   ├── manuscript.md
│   └── changelog.txt
├── results/
│   └── summarized_results.csv
└── src/
    ├── sightings_analysis.py
    └── runall.py
```

1. **Create a Project Directory**:
      - Put each project in its own directory, named after the project.

2. **Organize Text Documents**:
      - Place all text documents related to the project in a `doc` directory. This includes manuscripts, documentation, and electronic lab notebooks.

3. **Manage Data**:
      - Store raw data and metadata in a `data` directory.
      - Place files generated during cleanup and analysis in a `results` directory.

4. **Source Code Management**:
      - Keep all project source code in a `src` directory. This includes scripts and programs written for the project.

5. **Executable Programs**:
      - If applicable, store compiled programs in a `bin` directory.

6. **File Naming**:
      - Name all files to reflect their content or function. Avoid using sequential numbers or manuscript location references.

7. **Documentation and Metadata**:
      - Include a `README` file in the project's root directory to provide an overview of the project.
      - Add a `Makefile` to define the project's build pipeline.

8. **Version Control**:
      - Use a version control system to track changes and manage project versions.

9. **Backup and Synchronization**:
      - Ensure that the project directory is mirrored off the working machine using a system like Dropbox or a remote version control repository.

Here is another example of project template for a paper project:

```
paper/
├── analysis/
│   ├── README.md
│   ├── 00_clean_data.R
│   ├── 01_fit_models.R
│   ├── 02_make_figures.R
│   └── sandbox/
├── sim/
│   ├── README.md
│   ├── helper_functions.R
│   ├── sim_script.R
│   ├── run_sim_script.sh
│   └── sandbox/
├── figs/
├── notes/
├── ref_papers/
├── submitted/
├── revision/
├── final/
├── README.md
├── Makefile
├── my_paper.tex
└── my_refs.bib
```

## Data Management

Example of data analysis project:
```
analysis/
├── raw_data/
├── data/
├── R/
│   ├── 00_clean_data.R
│   ├── 01_fit_models.R
│   ├── 02_make_figures.R
│   ├── 03_summarize_results.R
│   └── 04_report.Rmd
├── figs/
├── sandbox/
│   └── exploratory.R
├── ref_papers/
├── Makefile
├── README.md
└── renv
```
Here is the guideline for data management:

1. **Save the Raw Data**:
      - Raw data are sacred... but may be a mess. Always keep the original data as it was first generated. 
     - Tempting to edit raw data by hand. <u>Don't!</u> Avoid overwriting raw data files with cleaned versions. Consider setting file permissions to read-only to prevent accidental changes.

2. **Backup Raw Data (3-2-1 Rule)**:
      - Keep at least 3 copies of your data
     - Store the copies on 2 different types of storage media
     - Keep 1 backup copy off-site (e.g., cloud storage, external drive at another location)

3. **Create Analysis-Friendly Data**:
      - Convert data into open, non-proprietary formats (e.g., CSV for tabular data). Use meaningful variable names and ensure filenames reflect their content.

4. **Everything scripted**:
      - Document Data Processing Steps. Record all steps used to process data. Write scripts for each stage of data processing to ensure reproducibility and ease of re-running analyses.
5. **Document Data**:
      - Use meta-data files to describe raw and cleaned data (e.g., `.csv` so easy to read)

## Collaboration

1. **Talk to your collaborators**:
      - Not everyone is comfortable with LaTeX or git or what we have learned in this course. So build an automated pipleline to generate other types of files, e.g., use `pandoc` to generate `docx` from `tex`.
      - Address organization from the outset.
      - Only have single master version online. Every collaborator's edit should be easily synchronized. Ideally, bring people on board to your (version controlled, reproducible) system. But try to design your project template robust for different systems. For example, if the master version is on overleaf, then design workflow compatible for all overleaf, dropbox, and git synchronization.
      - Make explicit decisions about how project members will communicate.
2. **Create an Overview of Your Project**
      - Have a `README` file in the project's home directory that explains the purpose of the project.
      - Include the project's title, a brief description, up-to-date contact information, and examples of how to run various tasks.
      - Include a `CONTRIBUTING` file that describes how to get the project running, use it, or contribute to it, including dependencies and tests.
3. **Create a Shared Document to Track Progress**
      - Maintain a plain text file like `log.md` or use project management platforms like [Trello](https://trello.com/) to create issues for each task.
      - Clearly describe tasks to make them understandable for newcomers.
      - Write the document, so we can backtrack of the progress of the project and how the project evolved.

