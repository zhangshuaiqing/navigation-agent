# Phase 2 实施计划 - 增强 Agent 智能

> **目标**: 提升 Agent 的导航策略和学习能力，充分利用 Phase 1 新增的环境功能
> **创建日期**: 2026-04-29
> **版本**: v0.4
> **状态**: 进行中

---

## 概述

Phase 1 完成了 GridWorld 环境的 4 个核心增强（局部视野、动态障碍物、多目标任务、随机起点/终点）。Phase 2 在此基础上，从 **记忆、策略、规划、学习** 四个维度全面提升 Agent 智能：

| 批号 | 功能 | 维度 | 优先级 | 依赖 |
|------|------|------|--------|------|
| A1 | 记忆模块 | 记忆 | P0 | 无 |
| A2 | ReAct Prompt 增强 | 策略 | P0 | 无 |
| A3 | 墙跟随 + 回溯 | 策略 | P0 | 无 |
| B1 | 多步规划 (Plan Tool) | 规划 | P1 | A1 |
| C1 | Q-Learning 基准 Agent | 学习 | P1 | A1 + A3 |

---

## 当前 Agent 现状与局限

### HeuristicNavigator

| 模式 | 策略 | 成功 | 局限 |
|------|------|------|------|
| `full` | BFS 最短路径 | ✅ 100% | 依赖全局信息，无学习能力 |
| `local` | 贪心 + 方向搜索 | ✅~95% | 死胡同/U型陷阱可能被困 |
| `fog_of_war` | 贪心 + 探索 | ✅~95% | 同上 |

### ReActNavigator

| 方面 | 现状 |
|------|------|
| Prompt | 极简（位置+目标+动作），不支持 multi-goal/local/dynamics |
| 每步成本 | 调 LLM，延迟 ~1-3s，无规划缓存 |
| 工具使用 | 4 tools，无 `plan_route`，每次从零推理 |

---

## 功能 A1：记忆模块

### 方案

新建 `src/agent/memory.py`，定义一个混入类，为所有 Agent 提供状态记忆能力。

```python
@dataclass
class AgentMemory:
    visited: Set[Tuple[int, int]]     # 所有访问过的位置
    trajectory: List[Tuple[int, int]] # 完整轨迹（位置序列）
    walls: Set[Tuple[int, int]]       # 探测到的障碍物
    goals_found: List[Tuple[int, int]] # 见过的目标位置
    forks: List[int]                  # trajectory 索引，指向分叉点
    last_actions: List[str]           # 最近 N 步的动作
```

**核心方法**：
- `update(obs)`：每步后更新记忆（visited, trajectory, walls, goals_found）
- `detect_fork(pos)`：检测当前位置是否是分叉点（≥2 个未访问方向）
- `backtrack()`：回溯到最近的分叉点，返回回溯路径上的动作序列
- `is_dead_end(pos)`：检查当前位置是否是死胡同（所有方向都访问过或被阻）

**Heuristic 集成**（`HeuristicNavigator.__init__` 新增 `use_memory=True`）：
- 默认 `use_memory=True`，可关闭
- `act()` 之前调用 `memory.update(obs)`
- local/fog 模式下，当贪心失败时，调用 `memory.backtrack()` 回溯

**ReAct 集成**：
- `_build_prompt()` 注入记忆摘要："已探索 N 个格子，探测到 M 个障碍物"
- 但不注入完整轨迹（token 开销太大）

### 改动范围

| 文件 | 行数 | 说明 |
|------|------|------|
| `src/agent/memory.py` | ~80 | AgentMemory 类（新文件） |
| `src/agent/__init__.py` | +1 | 导出 AgentMemory |
| `src/agent/navigator.py` | ~40 | HeuristicNavigator 集成 memory |

---

## 功能 A2：ReAct Agent Prompt 增强

### 方案

当前 prompt：
```
You are a navigation agent in a NxN grid world.
Your position: ..., Goal position: ...
Distance: ..., Valid moves: ...
```

增强后，`_build_prompt()` 根据环境参数动态生成内容。

**观察模式说明**：
- `full`：不加额外说明
- `local`：`[You are in LOCAL mode. You can only see {view_range} cells around you. Use sense_surroundings to explore. get_path_hint is disabled.]`
- `fog_of_war`：`[You are in FOG OF WAR mode. Unseen cells are hidden. Explore methodically.]`

**任务说明**（multi-goal）：
- `sequential`：`[Task: Visit goals in ORDER 1→2→3. Current: goal 1/3 at ({r},{c}). Use get_position to check progress.]`
- `any_order` / `collect`：同上但标注顺序不限

**动态障碍物**：
- 如有动态障碍物：`[Warning: {N} obstacles are moving. They may block paths unexpectedly.]`

**可用工具描述**（带当前状态的动态说明）：
```
Available tools:
- sense_surroundings: See what's around you
- move(direction): Move in a direction
- get_position: Check current position and goal status
- get_path_hint: [enabled in full mode] BFS shortest path
```

**策略建议**（根据场景切换）：
```
Strategy hints:
- full + single-goal: "Use get_path_hint for direct path, then move."
- full + multi-goal: "Complete goals in order. get_path_hint shows path to current goal."
- local: "Sense surroundings first, then move toward goal direction."
- fog_of_war: "Explore systematically. Mark visited areas mentally."
```

### 改动范围

| 文件 | 行数 | 说明 |
|------|------|------|
| `src/agent/navigator.py` | ~60 | `_build_prompt()` 重写 |

---

## 功能 A3：Heuristic Navigator 增强 - 墙跟随 + 回溯

### 方案

在 `HeuristicNavigator.act()` 中增加完整的决策树：

```
act(obs) → action:
  1. goal_visible?
     ✅ → BFS 最短路径 / 贪心接近

  2. goal_direction 方向可通行？
     ✅ → 向 goal_direction 移动

  3. 有未访问的邻居？
     ✅ → 选择最近的未访问格子（结合 memory）

  4. 死胡同检测 (is_dead_end)？
     ✅ → memory.backtrack() 回溯到分叉点

  5. 墙跟随 (右手法则)：
     - 当前方向右侧是否有障碍物？
       ✅ → 保持方向
     - 右侧无障 + 右侧格子未访问？
       ✅ → 右转并移动
     - 前方被堵？
       ✅ → 左转

  6. 最后的 fallback：
     随机选择一个合法动作
```

**右手法则说明**：一直贴着右侧墙壁走，能保证遍历整个连通区域（针对单连通图的最优探索策略）。

**回溯机制**：
- memory 记录分叉点（位置 + 当时有几个未访问分支）
- 进入死胡同时，沿 trajectory 逆向移动回到分叉点
- 重新选择另一个未访问分支

### 改动范围

| 文件 | 行数 | 说明 |
|------|------|------|
| `src/agent/navigator.py` | ~80 | HeuristicNavigator 完整决策树重写 |

---

## 功能 B1：多步规划 (Plan Tool)

### 方案

依赖：A1 记忆模块（提供局部地图）

**Heuristic 多步规划**：
- 不需要额外代码，`get_shortest_path()` 已经给了完整路径
- `act()` 返回路径中的下一步即可
- 在 multi-goal 场景下，当上一个 goal 完成后自动切换到下一个

**LLM Plan Tool（新增 tool）**：
```python
@tool
def plan_route(num_steps: int = 5) -> str:
    """Plan the next N steps using BFS on the observed map.
    
    Uses memory to build a local path from current position to goal.
    Returns a step-by-step action plan.
    """
```

- 在 `create_navigation_tools()` 中新增此 tool
- LLM 可以调用一次 `plan_route(5)` 获得接下来 5 步的建议，然后批量执行
- 减少 LLM 调用次数（从每步 1 次降低到每 5 步 1 次）

### 改动范围

| 文件 | 行数 | 说明 |
|------|------|------|
| `src/tools/navigation_tools.py` | ~25 | 新增 `plan_route` tool |
| `src/agent/navigator.py` | ~5 | ReAct 集成 |

---

## 功能 C1：Q-Learning 基准 Agent

### 方案

新建 `src/agent/q_learning.py`，实现一个基于 Q-Table 的强化学习 agent。

**状态编码**：
```
state = (
    agent_r,                    # agent 行坐标 (0..size-1)
    agent_c,                    # agent 列坐标
    goal_delta_r,               # goal 相对行偏移 (-size+1..size-1)
    goal_delta_c,               # goal 相对列偏移
    obstacle_surrounding,       # 周围 3x3 障碍物模式 (9-bit bitmask)
)
```

实际实现中为了控制 Q-table 大小，将连续坐标离散化或使用字典存储稀疏 Q-table。

**训练流程**：
```python
agent = QLearningAgent(env)
for episode in range(500):
    env.reset(new_map=True)
    while not done:
        state = agent.encode_state(obs)
        action = agent.epsilon_greedy(state)
        obs, reward, done, info = env.step(action)
        next_state = agent.encode_state(obs)
        agent.update(state, action, reward, next_state, done)
```

**epsilon 衰减**：从 1.0 线性衰减到 0.05，前 60% 训练轮次完成衰减。

**推理模式**：
- 训练完成后 `epsilon=0.0`，纯 greedy
- 可保存/加载 Q-table（pickle）

**性能预期**：
- 简单 6x6 网格：~200 个 episode 收敛
- 8x8 + 障碍物：~500 个 episode
- 完整训练后可达到 Heuristic BFS-level 的性能

### 改动范围

| 文件 | 行数 | 说明 |
|------|------|------|
| `src/agent/q_learning.py` | ~100 | QLearningAgent 类（新文件） |
| `src/agent/__init__.py` | +1 | 导出 QLearningAgent |
| `scripts/demo.py` | ~15 | `--agent qlearning` 支持 |
| `scripts/web_ui.py` | ~5 | Agent Type 增加 QLearning |
| `tests/test_agent.py` | ~30 | Q-Learning 基础测试 |

---

## 实施顺序

```
批次 A（P0，无依赖，可并行）:
  ├─ A1: 记忆模块        ~120行, 30min
  ├─ A2: ReAct Prompt增强 ~60行,  20min
  └─ A3: 墙跟随+回溯      ~80行,  30min

批次 B（P1，依赖 A1）:
  └─ B1: 多步规划         ~30行,  20min

批次 C（P1，依赖 A1 + A3）:
  └─ C1: Q-Learning       ~150行, 45min
```

每完成一个功能：
1. 单元测试验证
2. CLI Demo 快速验证
3. **提交等待用户验收**
4. 确认后推送 + 继续下一个

---

## 对现有系统的影响

| 组件 | 影响程度 | 说明 |
|------|---------|------|
| **CLI Demo** | 低 | `--agent` 可选值新增 `qlearning`; `--no-memory` 可关闭记忆 |
| **Web UI** | 低 | Agent Type radio 增加 "QLearning"（需要离线训练） |
| **LangChain Tools** | 低 | 新增 `plan_route` tool，不破坏现有 tools |
| **Heuristic Navigator** | 中 | `use_memory=True` 默认开启，可关闭回退到原行为 |
| **ReAct Navigator** | 低 | Prompt 变化仅影响新 episode |
| **Tests** | 中 | 新增 `tests/test_agent.py`（独立的 Agent 测试文件） |

---

## 验收标准

### A1：记忆模块
- [ ] local/fog 模式下 agent 不徘徊，能回溯出死胡同
- [ ] memory 正确记录 visited/walls/goals_found
- [ ] `use_memory=False` 时行为完全回退到 Phase 1

### A2：ReAct Prompt 增强
- [ ] multi-goal 场景下 prompt 包含任务进度说明
- [ ] local/fog 场景下 prompt 提示探索策略
- [ ] 动态障碍物场景下 prompt 包含 warning
- [ ] 非 full 模式 tools 描述标明禁用项

### A3：墙跟随 + 回溯
- [ ] U 型陷阱测试：agent 能回溯出来
- [ ] 墙跟随策略覆盖所有连通区域
- [ ] 回溯到分叉点后正确选择未访问分支

### B1：多步规划
- [ ] LLM 可通过 `plan_route(5)` 获得路径建议
- [ ] 单次 plan_route 调用可执行多步（减少 LLM 调用）

### C1：Q-Learning
- [ ] Q-table 随训练轮次收敛
- [ ] 训练后 agent 在 6x6 简单网格达到 100% 成功率
- [ ] 可保存/加载 Q-table

---

## 工作量预估

| 批次 | 功能 | 代码行数 | 预估时间 |
|------|------|---------|---------|
| A1 | 记忆模块 | ~120 行 | 30 min |
| A2 | ReAct Prompt 增强 | ~60 行 | 20 min |
| A3 | 墙跟随 + 回溯 | ~80 行 | 30 min |
| B1 | 多步规划 | ~30 行 | 20 min |
| C1 | Q-Learning | ~150 行 | 45 min |
| | **总计** | **~440 行** | **~2.5 hr** |
