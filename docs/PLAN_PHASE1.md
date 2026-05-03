# Phase 1 实施计划 - 增强环境功能

> **目标**: 使 GridWorld 更加丰富和实用
> **创建日期**: 2026-04-29
> **负责人**: Hermes Agent

---

## 概述

Phase 1 在现有 GridWorld + LangChain Agent 框架基础上，增加 4 个核心环境功能：

1. 随机起点/终点配置
2. 部分可观测状态（局部视野）
3. 动态障碍物（移动平台）
4. 多目标点任务系统

所有功能均通过新增参数实现，**默认行为完全向后兼容**，不影响现有 CLI Demo 和 Web UI。

---

## 功能 1：随机起点/终点配置

### 现状

- `__init__` 中起点固定为 `(0,0)`，终点固定为 `(size-1, size-1)`
- `reset()` 已支持自定义 `agent_pos` 和 `goal_pos`，但 `__init__` 未暴露随机配置选项

### 方案

1. `GridWorld.__init__` 新增参数：
   - `random_start_goal: bool = False`
   - `start_candidates: Optional[List[Tuple[int,int]]] = None`
   - `goal_candidates: Optional[List[Tuple[int,int]]] = None`

2. 当 `random_start_goal=True` 时：
   - 从网格四角/边缘随机选取起点和终点
   - 确保起点 ≠ 终点
   - 确保两者之间存在可达路径（BFS 验证，否则重新采样）

3. `reset()` 同步支持随机重采样

4. **Web UI**：增加 `Random Start/Goal` checkbox

### 改动范围

| 文件 | 行数 | 说明 |
|------|------|------|
| `src/env/gridworld.py` | ~25 | `__init__`, `_generate_map`, `reset` |
| `scripts/web_ui.py` | ~5 | 新增 checkbox |

### 兼容性

- 零破坏，纯新增参数，默认 `False`

---

## 功能 2：部分可观测状态（局部视野）

### 现状

- `_get_obs()` 返回 3x3 surroundings，但 agent 仍可通过 `get_position` 和 `get_path_hint` 获得全局信息
- 无 fog-of-war 机制，agent 理论上"全知"

### 方案

1. `GridWorld.__init__` 新增参数：
   - `observation_mode: str = "full"`  # 可选: "full" | "local" | "fog_of_war"
   - `view_range: int = 1`              # 视野半径，当前硬编码为 1

2. **三种观测模式**：

   - `"full"`（默认，当前行为）：
     - `sense_surroundings` 返回 3x3 信息
     - `get_path_hint` 返回全局 BFS 最短路径
     - `get_position` 返回精确坐标

   - `"local"`：
     - `sense_surroundings` 返回 `(2*view_range+1)^2` 的局部视野
     - `get_path_hint` 返回 `"视野受限，无法获取全局路径"`
     - `get_position` 仅返回相对方向（"Goal is to the south-east"），不返回精确坐标
     - Heuristic Navigator 退化为**贪心策略**（仅基于局部信息选择动作）

   - `"fog_of_war"`：
     - 维护 `visited_mask: np.ndarray` 记录 agent 历史访问过的格子
     - 已访问区域：显示真实内容
     - 未访问区域：显示 `"?"`
     - `sense_surroundings` 结合视野范围和 visited_mask
     - 渲染时未探索区域显示为灰色迷雾

3. **Heuristic Navigator 适配**：
   - `"full"` 模式：使用 BFS hint（当前行为）
   - `"local"/"fog_of_war"` 模式：使用贪心策略（选择使曼哈顿距离减小的动作，遇到障碍则沿墙跟随）

### 改动范围

| 文件 | 行数 | 说明 |
|------|------|------|
| `src/env/gridworld.py` | ~40 | `_get_obs`, 新增 `visited_mask`, 局部渲染 |
| `src/tools/navigation_tools.py` | ~15 | `get_path_hint`, `get_position` 增加模式判断 |
| `src/agent/navigator.py` | ~30 | Heuristic Navigator 增加局部导航模式 |
| `src/utils/viz.py` | ~10 | fog-of-war 渲染 |
| `scripts/web_ui.py` | ~10 | 模式选择下拉框 + 视野范围滑块 |

### 兼容性

- 默认 `"full"` 模式，与现有行为完全一致

---

## 功能 3：动态障碍物（移动平台）

### 现状

- 障碍物是静态的，`CellType.OBSTACLE = 1`
- 环境完全确定性，无时间维度变化

### 方案

1. 新增 `DynamicObstacle` 数据类：

   ```python
   @dataclass
   class DynamicObstacle:
       pos: Tuple[int, int]
       direction: str           # "up" | "down" | "left" | "right"
       speed: int = 1           # 每 N 个 agent 步移动一次
       move_prob: float = 0.5   # 每回合触发移动的概率
       boundary_mode: str = "bounce"  # "bounce" | "wrap" | "random"
   ```

2. `GridWorld.__init__` 新增参数：
   - `num_dynamic_obstacles: int = 0`
   - `dynamic_obstacle_speed: int = 1`

3. **动态更新逻辑**（在 `step()` 中 agent 移动后执行）：

   ```
   for each dynamic_obstacle:
       if step_count % speed == 0 and random() < move_prob:
           按 direction 移动 1 格
           如果越界:
               bounce 模式: 方向反转
               wrap 模式: 从对侧出现
               random 模式: 随机选择新方向
           如果目标位置是静态障碍物: 取消移动
   ```

4. **碰撞规则**：
   - Agent 移动到动态障碍物位置 → `reward = -1.0`，agent 被弹回原位
   - 动态障碍物移动到 agent 位置 → 同上
   - 动态障碍物之间不碰撞

5. **可达性保证**：
   - 动态障碍物初始位置不堵死唯一路径
   - 不保证运行时不堵死（增加环境不确定性）

6. **渲染**：
   - 新增 `CellType.DYNAMIC_OBSTACLE = 5`
   - 颜色：红色或橙色（区别于静态障碍物的深灰色）
   - Web UI 中显示移动方向箭头

### 改动范围

| 文件 | 行数 | 说明 |
|------|------|------|
| `src/env/gridworld.py` | ~80 | `DynamicObstacle` 类 + 动态更新 + 碰撞检测 |
| `src/utils/viz.py` | ~15 | 动态障碍物渲染 + 方向箭头 |
| `scripts/web_ui.py` | ~5 | 动态障碍物数量滑块 |
| `tests/test_env.py` | ~20 | 动态障碍物单元测试 |

### 兼容性

- 默认 `num_dynamic_obstacles=0`，不影响现有行为

---

## 功能 4：多目标点任务系统

### 现状

- 单一 `goal_pos`，到达即 `done=True`
- 无任务概念，无子目标

### 方案

1. 新增 `Task` 数据类：

   ```python
   @dataclass
   class Task:
       type: str                    # "sequential" | "any_order" | "collect"
       goals: List[Tuple[int, int]] # 目标点列表
       rewards: List[float]         # 每个子目标的奖励（可选，默认 uniform）
       completion_bonus: float = 5.0 # 全部完成的大奖
   ```

2. **三种任务模式**：

   - `"sequential"`（顺序访问）：
     - 必须按 `goals` 列表顺序访问
     - 激活索引 `active_idx`，初始为 0
     - 只有 `goals[active_idx]` 显示为激活状态（如闪烁或高亮）
     - 到达 `goals[active_idx]` → 小奖励 → `active_idx += 1`
     - 到达非激活目标 → 无奖励，不触发进展
     - 全部完成 → `done=True`，发放 `completion_bonus`

   - `"any_order"`（任意顺序）：
     - 访问所有 `goals` 即可，顺序不限
     - 维护 `completed_goals: Set[int]`
     - 每个新目标首次访问 → 小奖励
     - 全部访问 → `done=True`，发放 `completion_bonus`

   - `"collect"`（收集）：
     - 类似 `"any_order"`，但目标被访问后从地图上消失
     - 剩余目标数量实时减少
     - 适合"捡金币"类场景

3. `GridWorld.__init__` 新增参数：
   - `task: Optional[Task] = None`
   - `num_goals: int = 1`（快捷参数，自动生成随机目标点）

4. **BFS 路径规划适配**：
   - `get_shortest_path()` 返回到**当前激活目标**的最短路径
   - 顺序模式下，每完成一个子目标后重新计算

5. **Tools 适配**：
   - `get_position`：显示所有目标状态（已完成 ✅ / 激活 ⏳ / 未激活 ⚪）
   - `get_path_hint`：指向当前激活目标

6. **渲染**：
   - 不同状态目标使用不同颜色：
     - 激活目标：亮橙色 `#FF5722`（闪烁效果）
     - 未激活目标：暗橙色 `#BF360C`
     - 已完成目标：绿色 `#4CAF50`（打勾标记）
   - 收集模式下，已完成目标消失

### 改动范围

| 文件 | 行数 | 说明 |
|------|------|------|
| `src/env/gridworld.py` | ~100 | `Task` 类 + 多目标逻辑 + 奖励计算 |
| `src/tools/navigation_tools.py` | ~20 | `get_position`, `get_path_hint` 适配多目标 |
| `src/agent/navigator.py` | ~25 | Heuristic Navigator 适配多目标序列 |
| `src/utils/viz.py` | ~25 | 多目标渲染（不同颜色/状态） |
| `scripts/web_ui.py` | ~10 | 任务类型选择 + 目标数量滑块 |
| `tests/test_env.py` | ~25 | 多目标单元测试 |

### 兼容性

- 默认 `task=None`，单一 goal 行为完全不变
- `num_goals=1` 时等价于当前行为

---

## 实施顺序与依赖关系

```
功能 1 (随机起点/终点)
    ↓ 零依赖，可独立实现
功能 2 (局部视野)
    ↓ 零依赖，可独立实现
功能 3 (动态障碍物)
    ↓ 依赖功能 2（动态障碍物在视野边缘时只显示部分信息）
功能 4 (多目标任务)
    ↓ 依赖功能 1（随机起点/终点在多目标场景下更有用）
```

**推荐执行顺序**：

```
批次 A: 功能 1 + 功能 2（并行，互不依赖）
    ↓
批次 B: 功能 3（依赖功能 2）
    ↓
批次 C: 功能 4（依赖功能 1）
    ↓
批次 D: 文档更新 + 集成测试 + Web UI 统一更新
```

---

## 对现有系统的影响评估

| 组件 | 影响程度 | 说明 |
|------|---------|------|
| **CLI Demo** | 低 | 新增 `--obs-mode`, `--view-range`, `--dynamic`, `--task-type` 参数 |
| **Web UI** | 中 | 控制面板增加 6~8 个新控件，布局不变 |
| **LangChain Tools** | 低 | `get_path_hint`, `get_position` 增加条件判断，接口不变 |
| **Heuristic Agent** | 中 | 增加局部导航和多目标适配，默认行为不变 |
| **ReAct Agent** | 极低 | 通过 system prompt 感知新功能，核心逻辑不变 |
| **Tests** | 中 | 每个功能增加 2~3 个单元测试 |

---

## 工作量预估

| 功能 | 代码行数 | 预估时间 | 优先级 |
|------|---------|---------|--------|
| 功能 1 - 随机起点/终点 | ~30 | 15 min | P0 |
| 功能 2 - 局部视野 | ~60 | 30 min | P0 |
| 功能 3 - 动态障碍物 | ~100 | 45 min | P1 |
| 功能 4 - 多目标任务 | ~120 | 60 min | P1 |
| 文档 & 测试 | ~80 | 30 min | P2 |
| **总计** | **~390** | **~3 hr** | - |

---

## 验收标准

### 功能 1
- [ ] `GridWorld(random_start_goal=True)` 每次生成不同起点/终点
- [ ] BFS 验证可达性，不可达时自动重采样
- [ ] Web UI checkbox 可正常切换

### 功能 2
- [ ] `"local"` 模式下 `get_path_hint` 返回受限信息
- [ ] `"fog_of_war"` 模式下未访问区域渲染为灰色
- [ ] Heuristic Navigator 在局部模式下不使用 BFS hint 仍能完成导航
- [ ] Web UI 可切换观测模式

### 功能 3
- [ ] 动态障碍物按设定速度和概率移动
- [ ] Agent 与动态障碍物碰撞时获得负奖励并被弹回
- [ ] 边界反弹/环绕/随机转向模式正常工作
- [ ] Web UI 中动态障碍物显示方向箭头

### 功能 4
- [ ] `"sequential"` 模式必须按顺序访问目标
- [ ] `"any_order"` 模式顺序不限
- [ ] `"collect"` 模式下目标访问后消失
- [ ] 全部子目标完成时发放 completion_bonus
- [ ] Heuristic Navigator 正确追踪当前激活目标

---

## 后续关联

Phase 1 完成后，Phase 2（增强 Agent 智能）将基于新环境能力展开：

- Q-Learning / DQN 需要在局部视野和动态障碍物场景下训练
- 记忆模块在 fog-of-war 模式下尤为重要
- 多步规划在顺序任务场景下价值更大
