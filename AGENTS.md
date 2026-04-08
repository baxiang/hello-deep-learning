# AGENTS.md

## Project

Chinese learning notes for "Deep Learning from Scratch" (oreilly-japan/deep-learning-from-scratch). 27 Markdown notes + 11 Python code examples across 8 chapter folders.

## Structure

```
chXX-章节名/
├── 01-小节名.md
├── 02-小节名.md
└── code/
    └── 示例.py
```

Each chapter folder is independent. No shared modules between chapters.

## Rules for Editing

**Notes must follow the 6-section structure:**
1. 学习目标 (learning objectives)
2. 核心概念 (plain-language concepts, no formulas)
3. 原理解析 (deeper explanation)
4. 代码实战 (code with line-by-line explanations)
5. 避坑指南 (common beginner mistakes)
6. 课后思考 (exercises)

**Language:** Pure Chinese. No English technical terms without Chinese explanation.

**Code:** Each Python file must be independently runnable. No cross-chapter imports.

## Development Environment

### uv (Python Package Manager)

```bash
# Initialize environment
uv venv
source .venv/bin/activate
uv pip install -e ".[dev]"

# Install new package
uv pip install <package>

# Run code
python chXX-章节名/code/文件名.py
```

### VSCode + Jupyter

- Use `.venv/bin/python` as interpreter
- Install recommended extensions (see `.vscode/extensions.json`)
- Jupyter notebooks: select kernel → `.venv/bin/python`
- Format: Ruff auto-format on save

### Commands

- Run code: `python chXX-章节名/code/文件名.py`
- Start Jupyter: `jupyter notebook` or `jupyter lab`
- Format: `ruff format .`
- Lint: `ruff check .`

## Non-obvious Context

- Code uses only NumPy (no PyTorch/TensorFlow) — intentional for learning from scratch
- Chapter folders use Chinese names (e.g., `ch01-python基础/`)
- `.venv/` and `.ruff_cache/` are local dev artifacts (in `.gitignore`)
- No test framework, no lint CI — this is a notes repo, not software
- Python version: 3.13 (see `.python-version`)
