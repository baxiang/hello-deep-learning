# AGENTS.md

## Project

Chinese learning notes for "Deep Learning from Scratch" (oreilly-japan/deep-learning-from-scratch), now expanded with comprehensive LLM coverage from WeChat technical articles. 77 Markdown notes + 25 Python code examples across 12 chapter folders.

**Recent Expansion:** Added 8 new chapters and expanded 7 existing chapters covering LLM internals (Transformer, GPT, MoE, language generation, embeddings, attention mechanisms). All new content uses problem-driven approach with numerical examples and analogies.

## Structure

```
chXX-章节名/
├── 01-小节名.md
├── 02-小节名.md
└── code/
    └── 示例.py
```

Each chapter folder is independent. No shared modules between chapters.

**LLM-focused chapters:**
- ch06-学习技巧/05-残差连接.md (新增)
- ch10-注意力机制/06-位置嵌入.md, 07-GPT架构详解.md, 08-长文本处理.md, 09-MoE混合专家.md, 10-语言生成机制.md (新增)
- ch12-自然语言处理/00-嵌入基础.md, 06-子词分词器.md (新增)

## Rules for Editing Notes

**All notes must follow the 6-section structure:**
1. 学习目标 (learning objectives)
2. 开篇问题 (problem-driven - start with a question)
3. 核心概念 (plain-language concepts with numerical examples, no formulas initially)
4. 原理解析 (deeper explanation)
5. 代码实战 (code with line-by-line explanations)
6. 避坑指南 (common beginner mistakes)
7. 课后思考 (exercises)
8. 总结 (summary table)

**Writing style:**
- Problem-driven: Start with a question (e.g., "为什么必须有激活函数？")
- Numerical examples: Show specific calculations (e.g., "输入3 → 输出10 → 输出16")
- Analogies: Use real-world metaphors (e.g., "激活函数像音量旋钮")
- Comparison tables: Compare alternatives (e.g., "BatchNorm vs LayerNorm")
- Pure Chinese: No English technical terms without Chinese explanation

**Code:** Each Python file must be independently runnable. No cross-chapter imports. Pure NumPy implementation.

## Development Environment

### uv (Modern Python Package Manager)

```bash
# Install all dependencies (creates .venv automatically)
uv sync --all-groups

# Run code
uv run python chXX-章节名/code/文件名.py

# Add new package
uv add <package>

# Start Jupyter
uv run jupyter notebook
```

**Important:** Use `uv sync` / `uv add`, NOT `uv pip install`.

### VSCode + Jupyter

- Use `.venv/bin/python` as interpreter
- Install recommended extensions (see `.vscode/extensions.json`)
- Jupyter notebooks: select kernel → `.venv/bin/python`
- Format: Ruff auto-format on save

### Commands

- Run: `uv run python chXX-章节名/code/文件名.py`
- Jupyter: `uv run jupyter notebook`
- Format: `uv run ruff format .`
- Lint: `uv run ruff check .`

## Non-obvious Context

- Code uses only NumPy (no PyTorch/TensorFlow) — intentional for learning from scratch
- Chapter folders use Chinese names (e.g., `ch01-python基础/`)
- `.venv/`, `.uv-cache/`, `.ruff_cache/` are local dev artifacts (in `.gitignore`)
- `uv.lock` should be committed — it pins exact versions for reproducibility
- No test framework, no lint CI — this is a notes repo, not software
- Python version: 3.13 (see `.python-version`)
- `pyproject.toml` includes `playwright` dependency but all crawler scripts were deleted — only used temporarily for content gathering

## Recent Work Context (May 2026)

**LLM Content Integration:** Based on WeChat article series "中学生就能看懂从零开始理解LLM内部原理", created 8 new chapters and expanded 7 existing ones. Coverage now includes:

- Neural network basics → Language generation pipeline
- Embeddings (character → word → sub-word tokenization)
- Attention mechanisms (self-attention with numerical demos, multi-head with split dimension explanations)
- Transformer components (positional embeddings, residual connections, layer normalization)
- Advanced techniques (PI/YaRN for long context, MoE for efficiency, temperature parameters for generation control)

All new chapters follow: **问题驱动 → 数值演示 → 类比讲解 → 对比表格 → 代码实战**

## Verification Commands

Before committing new/edited chapters:
- `uv run python chXX-章节名/code/文件名.py` - verify code runs
- `uv run ruff check .` - lint check
- `uv run ruff format .` - format check
