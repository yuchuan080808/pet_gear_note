# Amazon Pet Content Pipeline

This project implements a drip-feed content pipeline for a Hugo/Astro static site about dog and cat supplies.

## Directory layout

```text
.
├── main.py                  # TaskManager, ScraperEngine, ContentGenerator, MarkdownExporter
├── requirements.txt          # Python dependencies
├── .env.example              # Copy to .env and set LLM_* values
├── .gitignore
├── data/my_output.json       # Amazon dog/cat category tree, committed to repo for Actions
├── data/tracking.json        # Git-trackable category status, created automatically
├── .autocli/adapters/        # Commit AutoCLI adapter YAML files here for GitHub Actions
├── .cache/bestsellers/       # Raw autocli bestseller JSON cache
├── .cache/products/          # Raw autocli product JSON cache
└── output/content/
    ├── dogs/                 # Generated dog Markdown files
    └── cats/                 # Generated cat Markdown files
```

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

Edit `.env` and set the Qwen/DashScope-compatible values:

```env
LLM_API_KEY=sk-your-dashscope-key-here
LLM_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
LLM_MODEL=qwen-plus
```

Later, switch to OpenAI or another OpenAI-compatible model by changing only these `LLM_*` values.

## Run

```bash
python main.py --batch-size 4
```

Useful options:

```bash
python main.py --batch-size 5 --top-n 20 --min-products 10
python main.py --reset-processing --batch-size 4
```

## Cron example

```cron
17 3 * * * cd /path/to/project && . .venv/bin/activate && python main.py --batch-size 4 >> logs/pipeline.log 2>&1
```

The generated Amazon links are constrained to clean `/dp/{ASIN}` URLs and the prompt explicitly forbids affiliate or tracking parameters.
Generated content also avoids exact prices. Product prices are converted to broad tiers before the LLM sees them, and the Markdown body is sanitized again after generation.

## GitHub Actions

The workflow at `.github/workflows/autopost.yml` supports manual runs and a weekly Monday 08:00 Asia/Shanghai cron run.

Before enabling it:

1. Commit `data/my_output.json` into the repository.
2. Run `autocli search amazon.com` locally, choose the Amazon adapter, then copy the generated YAML files from `~/.autocli/adapters/` into `.autocli/adapters/` in this repo.
3. The workflow downloads the latest Linux x86_64 AutoCLI release automatically and copies `.autocli/adapters/` into the runner's `~/.autocli/adapters/`.
4. Add repository secrets such as `LLM_API_KEY`, `LLM_BASE_URL`, and `LLM_MODEL`.
5. If your adapter command is not `autocli amazon bestsellers {url} -f json` and `autocli amazon product {asin} -f json`, add repository variables `AUTOCLI_BESTSELLERS_COMMAND` and `AUTOCLI_PRODUCT_COMMAND`.
6. Make sure the repository default branch is `main`, or adjust the final `git push` command.
