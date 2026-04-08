# VSCode + Jupyter + uv 配置指南

## 1. 安装必要工具

### 安装 uv（Python 包管理器）

```bash
# macOS
brew install uv

# 或
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### 安装 VSCode 扩展

打开 VSCode，会自动弹出推荐扩展列表。或者手动安装：

1. 按 `Cmd+Shift+X` 打开扩展面板
2. 搜索并安装以下扩展：
   - **ms-python.python** - Python 支持
   - **ms-toolsai.jupyter** - Jupyter Notebook 支持
   - **charliermarsh.ruff** - 代码格式化

## 2. 初始化项目环境

### 使用 uv 创建虚拟环境

```bash
# 进入项目目录
cd hello-deep-learning

# 创建虚拟环境（自动读取 .python-version）
uv venv

# 激活虚拟环境
source .venv/bin/activate

# 安装依赖
uv pip install -e ".[dev]"

# 或者分步安装
uv pip install numpy matplotlib jupyter ipykernel ruff
```

### 验证安装

```bash
# 检查 Python 版本
python --version

# 检查已安装的包
pip list

# 检查 Jupyter 内核
jupyter kernelspec list
```

## 3. 在 VSCode 中使用 Jupyter

### 创建新 Notebook

1. `Cmd+Shift+P` 打开命令面板
2. 输入 `Jupyter: Create New Jupyter Notebook`
3. 选择内核（选择 `.venv/bin/python`）
4. 保存文件到对应章节，如 `ch01-python基础/01-示例.ipynb`

### 运行现有 Notebook

1. 打开 `.ipynb` 文件
2. 右上角选择内核（`.venv/bin/python`）
3. 点击单元格旁的运行按钮，或按 `Shift+Enter`

### Notebook 快捷键

| 快捷键 | 功能 |
|--------|------|
| `Shift+Enter` | 运行当前单元格并跳到下一个 |
| `Ctrl+Enter` | 运行当前单元格 |
| `Alt+Enter` | 运行当前单元格并在下方插入新单元格 |
| `A` | 在上方插入单元格 |
| `B` | 在下方插入单元格 |
| `D,D` | 删除当前单元格 |
| `M` | 切换为 Markdown |
| `Y` | 切换为 Code |

## 4. 配置 Python 解释器

### 选择正确的解释器

1. `Cmd+Shift+P` → `Python: Select Interpreter`
2. 选择 `.venv/bin/python`
3. 如果没有显示，点击 `Enter interpreter path...` 手动输入：`${workspaceFolder}/.venv/bin/python`

### 验证配置

打开任意 Python 文件，查看 VSCode 右下角状态栏，应显示 `.venv` 环境。

## 5. 常见问题

### 内核找不到

```bash
# 重新注册内核
python -m ipykernel install --user --name hello-deep-learning --display-name "Python (hello-deep-learning)"
```

### 包导入错误

```bash
# 确保虚拟环境已激活
source .venv/bin/activate

# 重新安装依赖
uv pip install -e ".[dev]"
```

### Jupyter 服务启动失败

```bash
# 清除缓存
rm -rf .ipynb_checkpoints/
rm -rf .jupyter/

# 重启 VSCode
```

## 6. 快速命令参考

```bash
# 激活环境
source .venv/bin/activate

# 安装新包
uv pip install <package-name>

# 导出依赖
uv pip freeze > requirements.txt

# 运行代码
python ch01-python基础/code/文件名.py

# 启动 Jupyter Notebook
jupyter notebook

# 启动 Jupyter Lab
jupyter lab

# 格式化代码
ruff format .

# 检查代码质量
ruff check .
```
