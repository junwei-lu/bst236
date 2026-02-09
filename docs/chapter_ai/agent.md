# AI Agent


## What is an AI agent?

An **AI agent** is a goal-directed AI system that can do more than chat: it can **plan** a sequence of steps, **use tools** (like the terminal, editor, search, and linters/tests), observe the results, and **adapt** its next actions until the task is completed. In practice, that means you can give higher-level instructions (e.g., “add a feature,” “fix this bug,” “refactor this module”), and the agent can break the work into actions, make concrete changes, and verify them—acting like a collaborative assistant that operates directly on your project rather than only describing what you should do.



### IDE vs CLI Diagram


Command-line-inferface (CLI) coding becomes much more powerful when paired with an AI agent because the CLI is a **programmable interface** to your entire project and toolchain. Instead of giving one-off suggestions, an agent can execute multi-step workflows end-to-end—search the codebase, run scripts, inspect outputs, apply edits, and iterate until the result works—while keeping every action **explicit and reproducible** as commands you can re-run later. This tight loop makes it faster to debug, refactor, and validate changes (e.g., by running tests or linters immediately), and it scales well from small tasks to complex automation because the CLI provides consistent, composable primitives for building and verifying software.


Modern IDEs can do far more than autocomplete—debugging, refactors, builds, plugins—but a CLI agent shines at end-to-end orchestration.

| Dimension | IDE Paradigm  | CLI Paradigm |
|---|---|---|
| **Primary mode** | Interactive editing (you drive); tools assist inside the IDE | Delegation (agent drives); tools are invoked via commands |
| **Automation level** | Usually **single-step** help (edit, navigate, refactor one action at a time) | **Multi-step orchestration** (plan → run → check → fix → repeat) |
| **Execution** | Runs some tasks, but often you trigger and coordinate them manually | Directly runs shell commands/scripts; can chain many steps reliably |
| **Verification loop** | Helpful feedback, but results are often reviewed/assembled by the user | Built-in “prove it works” loop (tests, lint, logs, diffs, reruns) |
| **Best for** | Precise editing, local refactors, debugging in a rich UI | “Do this outcome end-to-end” (setup, data cleaning pipelines, releases, large changes) |




## Terminal-based Coding Tools

- [Claude Code](https://www.anthropic.com/claude-code): One of the best terminal coding tools. Cost $200 per month. You can define ToDo list and use agents for your tasks.
- [Codex](https://openai.com/api/codex/): OpenAI's AI agent for coding.

In our course, we will use a less popular CLI tool [Github Copilot CLI](https://github.com/features/copilot) for terminal coding to be consistent with our previous lectures on AI Copilot.


### Install Github Copilot CLI

Follow the instructions [here](https://github.com/github/copilot-cli?locale=en-US) to install Github Copilot CLI.

To install Copilot, 
on macOS or Linux, install with Homebrew:  
```bash
brew install copilot-cli
```

On Windows, use WinGet:  
```bash
winget install GitHub.Copilot
```

In the terminal, you can run `copilot` to see this:

![Github Copilot CLI](./agent.assets/copilot_cli.png)

Here are some commands for you to set up your Copilot CLI. We recommend you to read the official [tutorial](https://docs.github.com/en/copilot/how-tos/copilot-cli/cli-getting-started).

| Command     | Description                                           | Usage                                                                 |
|-------------|-------------------------------------------------------|----------------------------------------------------------------------|
| \login      | Login to your Github account                         | Type: \login                                                        |
| \logout     | Logout from your Github account                      | Type: \logout                                                       |
| \help       | Get help on using Copilot CLI                        | Type: \help what can you do?                                         |
| \version    | Get the version of Copilot CLI                       | Type: \version                                                        |
| \config     | Get the configuration of Copilot CLI                 | Type: \config                                                        |
| \set        | Set the configuration of Copilot CLI                 | Type: \set <key> <value>                                             |
| \model      | Set the model of Copilot CLI                         | Type: \model <model>                                                 |


### Agent Infrastructure


An AI agent pipeline architecture consists of three main components, each with a distinct role:

-   **Instructions**: The project-wide conventions. Instructions define project-wide conventions, policies, thresholds, and implementation details for agents and skills.

-   **Prompts**: The task runners. Prompts are simple, one-click triggers for executing a single, well-defined task. They are useful for automating repetitive actions.

-   **Agents**: The orchestrators. Agents manage complex, multi-step workflows. They can make decisions, use tools (like a terminal or file editor), and coordinate the overall process from start to finish.

-   **Skills**: The knowledge base. Skills provide detailed, domain-specific "how-to" documentation. An agent consults these skills to learn how to perform a specific task, such as calling a particular API or using a command-line tool.



This separation of concerns makes the system modular, scalable, and easy to maintain.

#### Directory Structure


Different CLI tools may have different requirements on the directory structure, but they are more or less similar. For VS Copilot and Copilot CLI, you can read the official [tutorial](https://code.visualstudio.com/docs/copilot/customization/overview) for more details. You can organize your components within a `.github/` directory in the root of your project:

```
.github/
├── agent/          # Contains all agent definitions
│   └── data-pipeline.agent.md
├── skills/         # Contains all skill categories
│   └── data-fetch/
│       └── SKILL.md
└── prompts/        # Contains all prompt definitions
    └── fetch-crypto.prompt.md
```

-   `.github/agent/`: Each file (e.g., `data-pipeline.agent.md`) defines an agent and its high-level workflow.
-   `.github/skills/`: Each sub-directory (e.g., `data-fetch/`) represents a domain of expertise, containing a `SKILL.md` file with implementation details.
-   `.github/prompts/`: Each file (e.g., `fetch-crypto.prompt.md`) defines a specific, single-shot task.

All components are defined in Markdown files with YAML frontmatter, making them easy for both humans and AI to read.


#### Instruction Files

Instruction files define project-wide conventions, policies, thresholds, and implementation details for agents and skills.

- **Location:**  
  - Typically at the root as `copilot-instructions.md`
  - Specific instructions with the name `<instruction-name>.instructions.md`
- **YAML Frontmatter:**  
  - Metadata such as `applyTo` to scope the rules (e.g., by file/glob).
- **Markdown Body:**  
  - Project-wide conventions, policies, thresholds, and reproducible implementation details for agents and skills.

**Example: `copilot-instructions.md`**
```markdown
---
applyTo: "**"
---
# Project-wide Instructions

- Use snake_case for Python filenames.
- Encoding: utf-8 for all file I/O.
- Volatility filter: |price_change_percentage_24h| > 5%.
- Outputs: crypto_raw.json, volatile_movers.json, volatility_report.json, daily_brief.txt, market_chart.png, index.html.
```
**Example: `doc.instructions.md`**
```markdown
---
applyTo: "docs/**/*.md"
---
# Documentation Standards

- Use sentence case for headings unless a proper noun is required.
- All code blocks must specify the language for syntax highlighting.
- Prefer active voice and concise sentences.
```



**Best Practices:**
- Keep instructions concise, clear, and definitive.
- Use `applyTo` to limit rules' scope when appropriate.
- Focus on policy and conventions that can be enacted or verified automatically—avoid including ad hoc commands.

#### Prompt Files

Prompt files define specific, one-click tasks for agents to execute.

- **Location:**  
  - `.github/prompts/<action>.prompt.md`
- **YAML Frontmatter:**  
  - `mode`: `agent` (for workflow triggers) or `manual`
  - `description`: brief summary of the prompt's purpose
  - `tools`: which tools to use, e.g., `terminalLastCommand`, `editFiles`, `codebase`
- **Markdown Body:**  
  - Bulletproof, step-by-step workflow.
  - Required input(s), specific commands/script invocations, and validation/checks (exit codes, file presence, minimal sanity checks).


The header is formatted as YAML frontmatter with the following fields:

| Field          | Description    |
|----------------|----------------|
| description    | A short description of the prompt. |
| name           | The name of the prompt, used after typing / in chat. If not specified, the file name is used. |
| argument-hint  | Optional hint text shown in the chat input field to guide users on how to interact with the prompt. |
| agent          | The agent used for running the prompt: ask, edit, agent, or the name of a custom agent. By default, the current agent is used. If tools are specified and the current agent is ask or edit, the default agent is agent. |
| model          | The language model used when running the prompt. If not specified, the currently selected model in model picker is used. |
| tools          | A list of tool or tool set names that are available for this prompt. Can include built-in tools, tool sets, MCP tools, or tools contributed by extensions. To include all tools of an MCP server, use the <server name>/* format. |
|                | Learn more about tools in chat. |


**Example: `fetch-crypto.prompt.md`**
```markdown
---
mode: agent
description: Fetch the top 50 cryptocurrencies from the CoinGecko API.
tools: [terminalLastCommand]

---
# Fetch Crypto Data

1. Use `curl` to fetch the current top 50 cryptocurrencies by market cap from the CoinGecko API.
2. Verify the output file `crypto_raw.json` exists and is not empty.
```

#### Agent Format 

An agent file defines the high-level plan.
The agent file is a markdown file with YAML frontmatter and a markdown body. The file name is `<name>.agent.md`.


-   **YAML Frontmatter**: `name`, `description`, and `tools` the agent can use.
-   **Markdown Body**: A step-by-step description of the workflow the agent should orchestrate.

**Example: `data-pipeline.agent.md`**
```markdown
---
name: Data Pipeline Agent
description: Orchestrates the entire crypto data pipeline.
tools: [terminalLastCommand, editFiles, codebase]
model: ['Claude Opus 4.5', 'GPT-4o']
---
# Crypto Data Pipeline Workflow

1.  **Fetch Data**: Use the `data-fetch` skill to get the top 50 coins from the CoinGecko API and save to `crypto_raw.json`.
2.  **Filter Data**: Use the `data-analysis` skill with `jq` to filter for volatile movers and save to `volatile_movers.json`.
3.  **Analyze Data**: Use the `data-analysis` skill with Python to generate a `volatility_report.json` and a `daily_brief.txt`.
4.  **Visualize Data**: Use the `data-viz` skill to create `market_chart.png`.
5.  **Publish**: Use the `github-pages` and `github-deploy` skills to build and deploy `index.html`.
```

#### Skill Format 

A skill file provides the "how-to" knowledge.
The skill file is a markdown file with YAML frontmatter and a markdown body. The file name is `<skill-name>/SKILL.md`.

-   **YAML Frontmatter**: `name` and `description` of the skill.
-   **Markdown Body**: Detailed instructions, code snippets, and best practices for a specific domain.

In practice, a `<skill-name>/` folder can contain more than just `SKILL.md`. These extra files help the agent **execute** the skill reliably and help humans **maintain** it over time:

-   **Scripts** (e.g., `run.sh`, `fetch.py`, `transform.R`): reusable “known-good” implementations the agent can run instead of re-creating commands every time.
-   **Templates** (e.g., `template.ipynb`, `report.md.tmpl`, `config.yaml.tmpl`): scaffolding the agent can copy/modify to produce consistent outputs.
-   **Example inputs/outputs** (e.g., `examples/`, small `sample.json`): concrete references for expected formats, useful for prompting and debugging.
-   **Validation fixtures** (e.g., `tests/`, `expected/`): lightweight checks (golden files, unit tests, sanity checks) so the agent can verify it did the right thing.
-   **Dependency notes** (e.g., `requirements.txt`, `environment.yml`, `Dockerfile`): pinning tools/packages needed for reproducible execution.
-   **Prompts / guidelines** (e.g., `PROMPT.md`, `CONVENTIONS.md`): house rules for style, edge cases, and safety constraints specific to that skill.

**Example: `data-fetch/SKILL.md`**
```markdown
---
name: Data Fetching
description: How to fetch data from the CoinGecko API.
---
## Fetching Market Data with `curl`

To get the top 50 cryptocurrencies, use the following `curl` command.

**Validation:**
- The command should exit with status 0.
- The output file `crypto_raw.json` should contain a JSON array of 50 objects.
```




## Case Study: The Crypto Watchtower

Let's see how these components work together in our crypto watchtower pipeline. The goal is to build a system that wakes up, fetches real-time cryptocurrency data from the [CoinGecko API](https://www.coingecko.com/), detects "Whale" movements (high volatility), generates a visual market report, and deploys a live dashboard to GitHub Pages—all without writing manual code.

#### The Core Concept:

Agents (Orchestrators): High-level managers that define what needs to be done.

Skills (Domain Modules): Specific technical instructions that define how to do it.

#### The Pipeline Architecture

**Tip:** You should always start with brainstorming with AI to design a prompt on how to design the agents, skills, and their interactions. Most of the markdown files could be generated by AI.

The data flows through our system in three distinct stages:

- **Ingest**: Fetch raw data -> Filter for volatility.

- **Analyze**: Calculate stats -> Generate charts.

- **Publish**: Build HTML -> Deploy to the Web.

**Key Thresholds:**

- **Volatility Filter**: Any coin moving > 5% (up or down) in 24h.

- **Whale Alert**: Any coin dropping > 10% triggers an ASCII warning.

### Task Decomposition

![Flow of Control](agent.assets/agent_workflow.png)

Decomposing the System
To build this complex system, we break it down into 3 Agents and 6 Skills.

**Agents:**

These are markdown files that contain the "Mission Objectives." They don't know the syntax of every tool, but they know which Skills to call.

**Skills:**

These are folders containing specific prompt context or scripts that the Agents use to execute tasks.

| Agent          | Skill            | Purpose                                    | Key Tools         |
|----------------|------------------|--------------------------------------------|-------------------|
| Data Pipeline  | data-fetch/      | Knowing how to query CoinGecko correctly.  | curl, API params  |
| Data Pipeline  | data-analysis/   | Filtering and computing stats.              | jq, pandas        |
| Data Pipeline  | data-viz/        | Creating beautiful charts.                  | matplotlib        |
| Code Quality   | code-review/     | Ensuring safety and best practices.         | linting           |
| Publish        | github-pages/    | Building the frontend dashboard.            | html, css         |
| Publish        | github-deploy/   | Handling git operations.                    | git               |

### Flow of Control

The flow of control is as follows:



1.  **Trigger**: The process starts by invoking the **Data Pipeline Agent**.
2.  **Orchestration**: The agent reads its instructions in `data-pipeline.agent.md`. The first step is "Fetch Data".
3.  **Skill Consultation**: To figure out *how* to fetch the data, the agent consults its knowledge base. It looks for the **`data-fetch` skill** and finds the `curl` command in its `SKILL.md` file.
4.  **Execution**: The agent uses its `terminal` tool to execute the `curl` command it just learned. The raw data is saved to `crypto_raw.json`.
5.  **Iteration**: The agent proceeds to the next step, "Filter Data". It consults the **`data-analysis` skill** to learn the correct `jq` command, executes it, and creates `volatile_movers.json`.
6.  **Completion**: This process repeats for all subsequent steps—analysis, visualization, and deployment. The agent acts as a project manager, consulting different skill experts at each stage to get the job done.

Here is the complete directory structure:

```
.github/
├─ agent/
│  ├─ code-quality.agent.md
│  ├─ data-pipeline.agent.md
│  └─ publisher.agent.md
├─ skills/
│  ├─ code-review/
│  │  └─ SKILL.md
│  ├─ code-revise/
│  │  └─ SKILL.md
│  ├─ data-analysis/
│  │  ├─ SKILL.md
│  │  └─ scripts/
│  │     └─ analyze_volatility.py
│  ├─ data-fetch/
│  │  ├─ SKILL.md
│  │  └─ scripts/
│  │     └─ fetch_top50.sh
│  ├─ data-viz/
│  │  ├─ SKILL.md
│  │  └─ scripts/
│  │     └─ generate_chart.py
│  ├─ github-action/
│  │  └─ SKILL.md
│  ├─ github-deploy/
│  │  └─ SKILL.md
│  └─ github-pages/
│     └─ SKILL.md
├─ prompts/
│  ├─ fetch-crypto.prompt.md
│  ├─ filter-volatile.prompt.md
│  ├─ generate-chart.prompt.md
│  ├─ build-dashboard.prompt.md
│  ├─ deploy-pages.prompt.md
│  └─ run-pipeline.prompt.md
└─ copilot-instructions.md
```