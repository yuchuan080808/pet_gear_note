# GitHub Actions Deployment

This repository is ready to run the drip-feed Amazon pet content pipeline in GitHub Actions.

## Files to commit

Commit these project files:

```text
main.py
requirements.txt
README.md
DEPLOYMENT.md
.env.example
.gitignore
.github/workflows/autopost.yml
.autocli/adapters/amazon/*.yaml
data/my_output.json
data/tracking.json
```

Do not commit `.env`, `.cache/`, `.data/`, `__pycache__/`, or local log files.

## GitHub secrets and variables

Add these in `Settings -> Secrets and variables -> Actions`.

Secrets:

```text
LLM_API_KEY
LLM_BASE_URL
LLM_MODEL
```

For Qwen / DashScope:

```text
LLM_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
LLM_MODEL=qwen-plus
```

Variables are optional. Add them only if your AutoCLI adapter command names differ from the defaults:

```text
AUTOCLI_BESTSELLERS_COMMAND=autocli amazon bestsellers {url} -f json
AUTOCLI_PRODUCT_COMMAND=autocli amazon product {asin} -f json
```

## Manual test

After pushing to GitHub:

1. Open the repository on GitHub.
2. Go to `Actions`.
3. Select `Auto Post Pet Content`.
4. Click `Run workflow`.

If it succeeds, the workflow will commit generated Markdown files under `content/` and update `data/tracking.json`.
