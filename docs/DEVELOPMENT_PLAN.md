# Navigation Agent - 开发计划

> **项目目标**: 使用 LangChain/LangGraph 构建 GridWorld 导航 Agent Demo
> **技术栈**: Python, LangChain, LangGraph, NumPy, uv
> **创建日期**: 2026-04-29

---

## 📋 项目概述

本项目实现了一个基于 LangChain 的导航 Agent，运行在自定义的 GridWorld 环境中。系统包含以下核心组件：

1. **GridWorld 环境**: NxN 网格世界，支持障碍物生成、BFS 路径规划、奖惩设计
2. **LangChain Tools**: sense_surroundings, move, get_position, get_path_hint 四个工具
3. **Agent 实现**: 支持 Heuristic/BFS 导航和 LLM-powered ReAct Agent 两种模式
4. **Demo & Benchmark**: 交互式演示和批量性能评测工具

---

## ✅ 已完成工作

### Phase 0 - 项目初始化 (2026-04-29)
- [x] 项目目录结构搭建 (src/{env,agent,tools,utils}, scripts, tests, docs)
- [x] uv 环境初始化，添加 langchain, langchain-openai, numpy, pydantic 依赖
- [x] GridWorld 环境实现（障碍物生成、BFS、路径检查、观察空间）
- [x] LangChain Tools 实现（4个工具）
- [x] Heuristic Navigator（基于 BFS hint 和目标寻找）
- [x] ReAct Navigator（基于 LangGraph 的 LLM Agent）
- [x] 可视化工具与 Demo 脚本
- [x] 单元测试

---

## 📝 待办事项（TODO）

### Phase 1 - 增强环境功能
**目标**: 使 GridWorld 更加丰富和实用
- [ ] 添加多个目标点和任务系统
- [ ] 添加动态障碍物/移动平台
- [ ] 添加部分可观测状态（仅能看到局部环境）
- [ ] 添加随机起点/终点配置

### Phase 2 - 增强 Agent 智能
**目标**: 提升 Agent 的导航策略和学习能力
- [ ] 实现 Q-Learning / DQN 基准 Agent
- [ ] 实现多步规划（不再每步都需要 LLM）
- [ ] 添加记忆模块（过往路径记忆）
- [ ] 增强 ReAct Agent 的 system prompt 和 few-shot 示例

### Phase 3 - 集成与部署
**目标**: 完善项目可用性
- [x] 添加 Gradio Web UI 可视化界面（网格渲染、路径箭头、状态面板、日志输出）
- [x] 支持多种 LLM 模型选择（gpt-4o-mini, gpt-4o, gpt-3.5-turbo）
- [ ] 支持多种 LLM Provider（Anthropic, local models via Ollama）
- [ ] 添加 Jupyter Notebook 教程
- [ ] 完善 CLI 参数和配置文件支持

### Phase 4 - 评估与分析
**目标**: 系统化评估各种 Agent 性能
- [ ] 构建标准化评估框架（成功率、步数、奖惩、时间）
- [ ] 不同网格大小和障碍物密度的敏感性分析
- [ ] LLM Agent vs Heuristic Agent 的对比分析
- [ ] 生成评估报告和可视化图表

### Phase 5 - 拓展应用
**目标**: 将框架拓展到更复杂的场景
- [ ] 支持连续空间（不再是离散网格）
- [ ] 支持多 Agent 协同导航
- [ ] 与具身智能模拟环境对接（如 Habitat, Isaac Sim）
- [ ] 支持图像/视觉观察输入

### Phase 6 - 优化与扩展
**目标**: 提升系统鲁棒性和可扩展性
- [ ] 代码覆盖率报告和更多单元测试
- [ ] 添加 CI/CD 配置
- [ ] 编写完整的使用文档和示例 Notebook
- [ ] 项目代码整理，准备开源（可选）

---

## 🔄 更新日志

| 日期 | 版本 | 更新内容 | 更新人 |
|------|------|----------|--------|
| 2026-04-29 | v0.1 | 项目初始化，完成 GridWorld 环境、LangChain Tools、Heuristic/ReAct Agent、Demo | Hermes |
| 2026-04-29 | v0.2 | 添加 Gradio Web UI，支持实时可视化、单步/自动运行、状态面板 | Hermes |

---

## 🎯 当前状态

**阶段**: Phase 3 部分完成
**下一步**: Phase 4 - 评估与分析 / Phase 1 增强环境功能
**阻塞项**: 无

---

## 💬 备注

- 项目使用 uv 管理 Python 依赖，所有操作通过 `uv run` 执行
- LLM Agent 需要设置 OPENAI_API_KEY 环境变量
- Heuristic Agent 在 8x8 网格上可达到 100% 成功率（使用 BFS hint）
- 项目路径: /media/zsq-508/data/project/navigation-agent
