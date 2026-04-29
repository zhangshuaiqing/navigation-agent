# Navigation Agent - GridWorld Demo

使用 LangChain/LangGraph 构建的 GridWorld 导航 Agent 项目。

## 项目概述

本项目实现了一个在网格世界中导航的智能体系统，支持：

- **GridWorld 环境**: 可配置的 NxN 网格，含障碍物生成、BFS 路径规划、距离奖励
- **Heuristic Agent**: 基于 BFS 的最短路径导航，无需 API Key
- **LLM ReAct Agent**: 基于 LangGraph 的大语言模型推理-行动 Agent
- **Web UI**: 基于 Gradio 的交互式可视化界面
- **CLI Demo**: 命令行演示与批量 Benchmark

## 快速开始

### 环境准备

项目使用 `uv` 管理 Python 依赖：

```bash
cd /media/zsq-508/data/project/navigation-agent

# 安装依赖（已包含在项目 pyproject.toml 中）
uv sync
```

### 运行 CLI Demo

```bash
# Heuristic Agent（默认）
uv run python scripts/demo.py

# 带渲染动画
uv run python scripts/demo.py --size 8 --seed 42 --delay 0.3

# Benchmark 20 轮
uv run python scripts/demo.py --benchmark 20 --no-render

# LLM ReAct Agent（需要 OPENAI_API_KEY）
export OPENAI_API_KEY="your-key"
uv run python scripts/demo.py --use-llm --llm-model gpt-4o-mini
```

### 启动 Web UI

```bash
uv run python scripts/web_ui.py
```

然后在浏览器打开 `http://localhost:7860`。

Web UI 支持：
- 实时网格可视化（Agent 路径、箭头轨迹）
- 单步执行 / 自动运行
- 可调节的网格大小、障碍物密度
- 一键切换 Heuristic / LLM Agent

### 运行测试

```bash
uv run python tests/test_env.py
```

## 项目结构

```
navigation-agent/
├── src/
│   ├── env/
│   │   └── gridworld.py          # GridWorld 环境
│   ├── agent/
│   │   └── navigator.py          # Heuristic + ReAct Agent
│   ├── tools/
│   │   └── navigation_tools.py   # LangChain Tools
│   └── utils/
│       └── viz.py                # 渲染 & Benchmark
├── scripts/
│   ├── demo.py                   # CLI Demo
│   └── web_ui.py                 # Gradio Web UI
├── tests/
│   └── test_env.py               # 单元测试
├── docs/
│   ├── DEVELOPMENT_PLAN.md       # 开发计划
│   └── WEB_UI.md                 # Web UI 详细文档
└── pyproject.toml
```

## Agent 对比

| 特性 | Heuristic Navigator | ReAct Navigator |
|------|---------------------|-----------------|
| 依赖 | 无需 API Key | 需要 OPENAI_API_KEY |
| 策略 | BFS 最短路径 | LLM 推理 + 工具调用 |
| 速度 | 极快 | 受 LLM API 延迟限制 |
| 成功率 | 100%（已知地图） | 取决于 LLM 推理能力 |
| 适用场景 | 快速验证、Benchmark | 复杂推理、少样本学习 |

## 开发计划

详见 [docs/DEVELOPMENT_PLAN.md](docs/DEVELOPMENT_PLAN.md)

当前已完成 Phase 0 ~ Phase 3（Web UI）。

## License

MIT
