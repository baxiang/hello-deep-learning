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

## Commands

- Run a code file: `python chXX-章节名/code/文件名.py`
- Python dependencies: `pip install numpy matplotlib`

## Non-obvious Context

- Code examples use only NumPy (no PyTorch/TensorFlow). This is intentional — the book teaches from scratch.
- Chapter numbering uses Chinese characters in folder names (e.g., `ch01-python基础/`).
- `.venv/` and `.ruff_cache/` are local dev artifacts, already in `.gitignore`.
- No test framework, no lint config. This is a notes repo, not a software project.
