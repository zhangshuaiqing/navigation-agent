# Web UI 使用文档

## 概述

Navigation Agent Web UI 是一个基于 Gradio 的交互式可视化界面，用于在浏览器中实时观察和控制 GridWorld 导航 Agent。

## 启动方式

```bash
cd /media/zsq-508/data/project/navigation-agent
uv run python scripts/web_ui.py
```

启动后会显示访问地址：

```
Running on local URL:  http://0.0.0.0:7860
```

在浏览器中打开该地址即可使用。

## 界面说明

### 左侧面板 - 控制区

| 控件 | 说明 |
|------|------|
| **Grid Size** | 网格大小（4 ~ 20） |
| **Obstacle Ratio** | 障碍物比例（0.0 ~ 0.5） |
| **Random Seed** | 随机种子（0 表示随机） |
| **Agent Type** | `heuristic`（BFS）或 `llm`（ReAct） |
| **LLM Model** | LLM Agent 时使用的模型 |
| **Auto-run Delay** | 自动运行时每步间隔（秒） |
| **Reset** | 重置环境和 Agent |
| **Step** | 执行单步 |
| **Run** | 自动运行直到完成 |
| **Stop** | 停止自动运行 |

### 右侧面板 - 可视化区

| 区域 | 说明 |
|------|------|
| **Grid World** | 网格可视化图像 |
| **Status** | 当前状态（位置、步数、奖励等） |
| **Logs** | 操作日志（保留最近 50 条） |

### 图例

| 符号 | 颜色 | 含义 |
|------|------|------|
| A | 绿色 | Agent 当前位置 |
| G | 橙色 | 目标位置 |
| 蓝色箭头 | - | Agent 走过的路径 |
| 浅蓝色 | - | 已访问的格子 |
| 深灰色 | - | 障碍物 |
| 浅灰色 | - | 空格 |

## 使用流程

### 1. 基础导航（Heuristic Agent）

1. 设置网格大小（推荐 6 ~ 10）
2. 点击 **Reset** 生成环境
3. 点击 **Run** 观察 Agent 自动导航到目标
4. 或点击 **Step** 逐步执行

### 2. LLM Agent 体验

1. 选择 **Agent Type** 为 `llm`
2. 选择 **LLM Model**（推荐 `gpt-4o-mini`，速度快且便宜）
3. 确保已设置环境变量 `OPENAI_API_KEY`
4. 点击 **Reset**，然后 **Run**

> 注意：LLM Agent 每步都需要调用 OpenAI API，运行速度较慢。

### 3. 自定义挑战

- 增大 **Grid Size** 到 15+ 增加难度
- 提高 **Obstacle Ratio** 到 0.3+ 增加障碍物密度
- 使用不同 **Random Seed** 生成不同地图
- 尝试不使用 BFS hint 的纯 LLM 推理（在代码中设置 `use_bfs_hint=False`）

## 架构

Web UI 的核心是 `AppState` 类，它维护：

- `env`: GridWorld 环境实例
- `agent`: 当前 Agent 实例（HeuristicNavigator 或 ReActNavigator）
- `path_history`: Agent 走过的路径记录
- `step_count` / `total_reward`: 统计信息
- `is_running`: 自动运行状态标志

Gradio 通过事件绑定将按钮点击映射到 `AppState` 的方法，实现状态更新和界面刷新。

## 故障排除

| 问题 | 解决方案 |
|------|----------|
| `OPENAI_API_KEY not set` | 设置环境变量后重启 Web UI |
| 网格不显示 | 点击 **Reset** 初始化环境 |
| Run 按钮无响应 | 检查是否已经到达目标或超过最大步数 |
| 页面加载慢 | 减小 Grid Size 或增大 Auto-run Delay |

## API Key 配置

ReAct (LLM) Agent 需要 OpenAI API Key。配置方式：

### 临时设置
```bash
export OPENAI_API_KEY="sk-your-api-key"
uv run python scripts/demo.py --use-llm
```

### .env 文件（推荐）
```bash
# OpenAI
echo 'OPENAI_API_KEY="sk-your-key"' >> .env

# DeepSeek
echo 'DEEPSEEK_API_KEY="sk-your-key"' >> .env

# Kimi (Moonshot)
echo 'MOONSHOT_API_KEY="sk-your-key"' >> .env
```
`.env` 已在 `.gitignore` 中，不会提交到仓库。

### 获取 API Key
访问 https://platform.openai.com/api-keys 创建 key。
推荐使用 `gpt-4o-mini` 模型（低成本、速度快）。

## 本地模型配置

### Ollama (推荐，最简单)
确保已安装 [Ollama](https://ollama.com) 并启动后：

```bash
# 拉取模型（任选一个）
ollama pull qwen2.5:7b       # 7B 参数，速度快
ollama pull llama3.2:3b      # 更轻量

# 无需额外配置，直接使用
uv run python scripts/demo.py --use-llm --llm-provider ollama --llm-model qwen2.5:7b
```

默认 Ollama API 地址为 `http://localhost:11434/v1`。
如果 Ollama 运行在非默认端口：
```bash
export OLLAMA_BASE_URL="http://your-host:11434/v1"
```

### vLLM (高性能推理)
确保已安装 [vLLM](https://docs.vllm.ai) 并启动服务：

```bash
# 启动 vLLM 服务
vllm serve Qwen/Qwen2.5-7B-Instruct --port 8000

# 连接使用
uv run python scripts/demo.py --use-llm --llm-provider vllm

# 自定义地址
export VLLM_BASE_URL="http://your-server:8000/v1"
```

### 自定义 OpenAI 兼容 API
对接任何兼容 OpenAI API 格式的本地/远程服务：

```bash
# 设置环境变量
export CUSTOM_LLM_BASE_URL="http://your-endpoint:port/v1"
export CUSTOM_LLM_MODEL="your-model-name"
export CUSTOM_LLM_API_KEY="sk-..."  # 可选，无认证可跳过

# 使用
uv run python scripts/demo.py --use-llm --llm-provider custom
```

支持的服务包括：
- **vLLM** (本地推理)
- **llama.cpp / llama-server** (本地推理)
- **Xinference** (本地推理)
- **TensorRT-LLM** (本地推理)
- **任何 OpenAI 兼容的 API 代理**

### 推荐模型

| 模型 | 适合导航任务 | 参数 | Provider |
|------|-------------|------|----------|
| `qwen2.5:7b` | ⭐⭐⭐ | 7B | ollama |
| `llama3.2:3b` | ⭐⭐ | 3B | ollama |
| `Qwen2.5-7B-Instruct` | ⭐⭐⭐ | 7B | vllm |
| `gpt-4o-mini` | ⭐⭐⭐ | - | openai |

## 扩展建议

- 添加键盘快捷键（方向键控制 Agent）
- 添加保存/加载地图功能
- 添加多 Agent 同时运行对比
- 添加 GIF 录制功能
