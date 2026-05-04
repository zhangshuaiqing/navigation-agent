"""
Navigation agents for GridWorld using LangChain/LangGraph.
"""

import os
from typing import List, Optional, Dict, Any

from langchain_core.language_models import BaseChatModel
from langchain_core.tools import BaseTool
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent

from ..env.gridworld import GridWorld
from ..tools.navigation_tools import create_navigation_tools
from .memory import AgentMemory


# ═══════════════════════════════════════════════════════════════
# LLM Provider configuration
# ═══════════════════════════════════════════════════════════════

LLM_PROVIDERS = {
    "openai": {
        "env_key": "OPENAI_API_KEY",
        "class": "langchain_openai.ChatOpenAI",
        "default_model": "gpt-4o-mini",
        "base_url": None,
    },
    "deepseek": {
        "env_key": "DEEPSEEK_API_KEY",
        "class": "langchain_openai.ChatOpenAI",
        "default_model": "deepseek-chat",
        "base_url": "https://api.deepseek.com",
    },
    "kimi": {
        "env_key": "MOONSHOT_API_KEY",
        "class": "langchain_openai.ChatOpenAI",
        "default_model": "moonshot-v1-8k",
        "base_url": "https://api.moonshot.cn/v1",
    },
}


def get_llm(
    provider: str = "openai",
    model: Optional[str] = None,
    temperature: float = 0,
) -> BaseChatModel:
    """Create an LLM instance from the specified provider.
    
    Args:
        provider: One of 'openai', 'deepseek', 'kimi'
        model: Model name (defaults to provider's default)
        temperature: LLM temperature
    
    Returns:
        A ChatOpenAI-compatible LLM instance
    
    Raises:
        ValueError: If provider is unknown or API key is not set
    """
    if provider not in LLM_PROVIDERS:
        raise ValueError(
            f"Unknown provider: {provider}. "
            f"Available: {', '.join(LLM_PROVIDERS.keys())}"
        )
    
    cfg = LLM_PROVIDERS[provider]
    api_key = os.environ.get(cfg["env_key"])
    if not api_key:
        raise ValueError(
            f"{cfg['env_key']} not set. "
            f"Set the environment variable '{cfg['env_key']}' "
            f"or provide an LLM instance directly."
        )
    
    kwargs = {
        "model": model or cfg["default_model"],
        "temperature": temperature,
        "api_key": api_key,
    }
    if cfg["base_url"]:
        kwargs["base_url"] = cfg["base_url"]
    
    return ChatOpenAI(**kwargs)


class NavigationAgent:
    """Base class for navigation agents."""
    
    def __init__(self, env: GridWorld):
        self.env = env
        self.tools = create_navigation_tools(env)
        self.history: List[Dict[str, Any]] = []
    
    def act(self, observation: Optional[Dict] = None) -> str:
        """Take an action based on observation. Returns action name."""
        raise NotImplementedError
    
    def post_step(self):
        """Called after env.step() to update agent state (e.g. memory)."""
        pass
    
    def reset(self):
        """Reset agent state."""
        self.history.clear()


class HeuristicNavigator(NavigationAgent):
    """
    A heuristic-based navigator that doesn't require an LLM.
    Uses BFS, wall-following, dead-end backtracking, and memory.
    """
    
    def __init__(self, env: GridWorld, use_bfs_hint: bool = True, use_memory: bool = True):
        super().__init__(env)
        self.use_bfs_hint = use_bfs_hint
        self.use_memory = use_memory
        self.memory = AgentMemory()
        self.last_positions: List[tuple] = []
        self.max_history = 10
    
    def act(self, observation: Optional[Dict] = None) -> str:
        """Take action using heuristic."""
        obs = observation or self.env._get_obs()
        
        # In full mode with BFS hint, use shortest path
        if self.env.observation_mode == "full" and self.use_bfs_hint:
            path = self.env.get_shortest_path()
            if path and len(path) > 1:
                next_step = path[1]
                current = self.env.agent_pos
                dr = next_step[0] - current[0]
                dc = next_step[1] - current[1]
                
                if dr == -1:
                    action = "up"
                elif dr == 1:
                    action = "down"
                elif dc == -1:
                    action = "left"
                else:
                    action = "right"
                
                self.history.append({
                    "action": action,
                    "reason": f"BFS path hint: moving toward {next_step}"
                })
                return action
        
        # Local/fog mode or BFS disabled: use greedy goal-seeking + exploration
        ar, ac = obs["agent_pos"]
        gr, gc = obs.get("goal_pos") or self.env.goal_pos
        goal_visible = obs.get("goal_visible", True)
        goal_direction = obs.get("goal_direction")
        
        valid = self.env.get_valid_actions()
        
        def _score_action(action):
            dr, dc = self.env.DIRECTIONS[action]
            nr, nc = ar + dr, ac + dc
            score = 0.0
            
            # Prefer moves toward goal direction even when not visible
            if goal_direction and not goal_visible:
                direction_bonus = 0.0
                if "N" in goal_direction and dr < 0:
                    direction_bonus += 3.0
                if "S" in goal_direction and dr > 0:
                    direction_bonus += 3.0
                if "W" in goal_direction and dc < 0:
                    direction_bonus += 3.0
                if "E" in goal_direction and dc > 0:
                    direction_bonus += 3.0
                score -= direction_bonus
            
            if goal_visible:
                score += abs(nr - gr) + abs(nc - gc)
            else:
                if self.use_memory:
                    # Use memory to prefer unvisited cells
                    if (nr, nc) not in self.memory.visited:
                        score -= 5.0
                    else:
                        score += 1.0
                else:
                    # Fallback to env visited_mask
                    if not self.env.visited_mask[nr, nc]:
                        score -= 5.0
                    else:
                        score += 1.0
            
            # Strong penalty for revisiting very recent positions
            if (nr, nc) in self.last_positions[-3:]:
                score += 20.0
            elif (nr, nc) in self.last_positions:
                score += 5.0
            
            return score
        
        # Try 1: Goal visible or goal direction available
        best_action = None
        best_score = float('inf')
        
        for action in valid:
            score = _score_action(action)
            if score < best_score:
                best_score = score
                best_action = action
        
        # Try 2: Dead-end backtracking (using memory)
        if self.use_memory and best_action and best_score >= 10.0:
            # All moves are bad (penalized as visited) - check for dead end
            if self.memory.is_dead_end((ar, ac), self.env.size, valid):
                backtrack_action = self.memory.backtrack((ar, ac))
                if backtrack_action and backtrack_action in valid:
                    reason = f"Dead-end backtrack to {self.memory.trajectory[-2] if len(self.memory.trajectory) >= 2 else 'previous'}"
                    self.history.append({"action": backtrack_action, "reason": reason})
                    self.last_positions.append((ar, ac))
                    if len(self.last_positions) > self.max_history:
                        self.last_positions.pop(0)
                    return backtrack_action
        
        # Try 3: If all directions penalized, find any unvisited direction
        if best_action is None or (best_score >= 10.0 and self.use_memory):
            unvisited = self.memory.find_unvisited_direction((ar, ac), self.env.size, valid)
            if unvisited:
                best_action = unvisited
                best_score = 0.0
        
        if best_action is None and valid:
            best_action = valid[0]
        
        # Record fork points for future backtracking
        if self.use_memory:
            if self.memory.detect_fork((ar, ac), self.env.size, valid):
                self.memory.record_fork((ar, ac), self.env.size, valid)
        
        self.last_positions.append(self.env.agent_pos)
        if len(self.last_positions) > self.max_history:
            self.last_positions.pop(0)
        
        reason = (
            f"Goal-seeking: chosen to minimize distance ({best_score:.1f})"
            if goal_visible
            else f"Directional exploration: heading {goal_direction} (score {best_score:.1f})"
        )
        self.history.append({
            "action": best_action or "up",
            "reason": reason
        })
        
        return best_action or "up"
    
    def reset(self):
        super().reset()
        self.last_positions.clear()
        self.memory.reset()
        # Record initial position
        self.post_step()
    
    def post_step(self):
        """Update memory with post-move state."""
        if not self.use_memory:
            return
        obs = self.env._get_obs()
        self.memory.update(
            pos=self.env.agent_pos,
            obs=obs,
            actions=self.memory.last_actions,
            grid_size=self.env.size,
        )


class ReActNavigator(NavigationAgent):
    """
    A ReAct navigator powered by an LLM via LangGraph.
    """
    
    def __init__(
        self,
        env: GridWorld,
        llm: Optional[BaseChatModel] = None,
        llm_provider: str = "openai",
        llm_model: Optional[str] = None,
        max_iterations: int = 50,
    ):
        super().__init__(env)
        self.max_iterations = max_iterations
        
        if llm is None:
            try:
                llm = get_llm(provider=llm_provider, model=llm_model)
            except ValueError as e:
                raise ValueError(
                    f"Failed to create LLM for provider '{llm_provider}': {e}\n"
                    f"Either provide an 'llm' instance directly, "
                    f"or set the appropriate environment variable."
                )
        
        self.llm = llm
        self.agent = create_react_agent(
            model=llm,
            tools=self.tools,
        )
        self.messages: List[Any] = []
        # Store provider info for display
        self.llm_provider = llm_provider
        self.llm_model = llm.model_name if hasattr(llm, 'model_name') else str(llm.model)
    
    def act(self, observation: Optional[Dict] = None) -> str:
        """
        Run the ReAct agent for one episode or step.
        
        For gridworld, we prompt the agent to navigate and let it
        decide when to move.
        """
        prompt = self._build_prompt()
        
        result = self.agent.invoke({
            "messages": self.messages + [HumanMessage(content=prompt)]
        })
        
        self.messages = result["messages"]
        
        # Extract the last tool call for 'move'
        last_move = None
        for msg in reversed(self.messages):
            if isinstance(msg, AIMessage) and msg.tool_calls:
                for tc in msg.tool_calls:
                    if tc["name"] == "move":
                        last_move = tc["args"].get("direction", "")
                        break
            if last_move:
                break
        
        if last_move:
            self.history.append({
                "action": last_move,
                "reason": "LLM ReAct decision"
            })
            return last_move
        
        # Fallback: no move made, try to get a valid action
        valid = self.env.get_valid_actions()
        fallback = valid[0] if valid else "up"
        self.history.append({
            "action": fallback,
            "reason": "LLM did not produce move, using fallback"
        })
        return fallback
    
    def _build_prompt(self) -> str:
        """Build the navigation prompt for the LLM, adapting to environment settings."""
        env = self.env
        obs = env._get_obs()
        ar, ac = obs["agent_pos"]
        goal_pos = obs.get("goal_pos")
        gr, gc = goal_pos if goal_pos is not None else (None, None)
        valid = env.get_valid_actions()
        mode = env.observation_mode
        size = env.size
        
        lines = []
        lines.append(f"You are a navigation agent in a {size}x{size} grid world.")
        lines.append(f"Your current position: ({ar}, {ac})")
        
        # ── Goal / Task info ─────────────────────────────
        task = obs.get("task")
        if task:
            total_goals = task["total_goals"]
            completed = task["completed"]
            task_type = task["type"]
            current_idx = task["current_goal_index"]
            type_desc = {
                "sequential": "Visit goals IN ORDER",
                "any_order": "Visit all goals in ANY order",
                "collect": "Collect all goals (they disappear after collection)",
            }
            lines.append(f"Task: {type_desc.get(task_type, task_type)} | Goals: {completed}/{total_goals} completed")
            lines.append(f"Current goal #{current_idx + 1}/{total_goals}: {obs.get('goal_pos', 'UNKNOWN')}")
        elif gr is not None:
            lines.append(f"Goal position: ({gr}, {gc})")
            lines.append(f"Distance to goal: {obs['distance_to_goal']}")
        else:
            lines.append(f"Goal position: NOT IN VIEW (explore to find it)")
        
        # ── Observation mode warning ─────────────────────
        if mode == "local":
            vr = obs["view_range"]
            direction = obs.get("goal_direction")
            lines.append("")
            lines.append(f"[LOCAL MODE] You can only see {2*vr+1}x{2*vr+1} cells around you.")
            lines.append(f"[LOCAL MODE] Goal direction hint: {direction or 'UNKNOWN'}")
            lines.append("[LOCAL MODE] Use sense_surroundings frequently. get_path_hint is DISABLED.")
        elif mode == "fog_of_war":
            vr = obs["view_range"]
            direction = obs.get("goal_direction")
            lines.append("")
            lines.append(f"[FOG OF WAR] Only visited cells are visible. Unvisited cells show as '?'.")
            lines.append(f"[FOG OF WAR] Goal direction hint: {direction or 'UNKNOWN'}")
            lines.append("[FOG OF WAR] Explore methodically. get_path_hint is DISABLED.")
        
        # ── Dynamic obstacles warning ────────────────────
        if env.num_dynamic_obstacles > 0 and env.dynamic_obstacles:
            lines.append("")
            lines.append(f"[WARNING] {len(env.dynamic_obstacles)} obstacles are MOVING. Paths may change.")
        
        # ── Multi-target awareness ───────────────────────
        if task and total_goals > 1:
            lines.append("")
            lines.append("[MULTI-GOAL] Use get_position to check which goals are done/pending/active.")
        
        # ── Valid moves ──────────────────────────────────
        lines.append("")
        lines.append(f"Valid moves from ({ar}, {ac}): {valid}")
        
        # ── Tools ────────────────────────────────────────
        lines.append("")
        lines.append("Available tools:")
        lines.append("- sense_surroundings: See what's around you (always available)")
        lines.append("- move(direction): Move in one of the valid directions")
        lines.append("- get_position: Check current position and goal progress")
        if mode == "full":
            lines.append("- get_path_hint: BFS shortest path to current goal")
        else:
            lines.append("- get_path_hint: [DISABLED in " + mode + " mode]")
        
        # ── Strategy hints (scenario-specific) ───────────
        lines.append("")
        lines.append("Strategy:")
        if mode == "full" and (not task or total_goals == 1):
            lines.append("1. Use get_path_hint to get the optimal path")
            lines.append("2. move() in the suggested direction")
            lines.append("3. Repeat until goal reached")
        elif mode == "full" and total_goals > 1 and task_type == "sequential":
            lines.append("1. Use get_path_hint to get path to current goal")
            lines.append("2. move() toward it. After reaching a goal, get_path_hint points to the next one.")
            lines.append(f"3. Complete all {total_goals} goals in order.")
        elif mode in ("local", "fog_of_war"):
            lines.append("1. Start with sense_surroundings to see nearby cells")
            lines.append("2. Move toward the goal direction hint")
            lines.append("3. If blocked, explore alternate paths")
            lines.append("4. Use sense_surroundings after each move to update your view")
        
        lines.append("")
        lines.append("Think step by step. Make ONE move at a time.")
        
        return "\n".join(lines)
    
    def reset(self):
        super().reset()
        self.messages.clear()


def create_react_navigator(
    env: GridWorld,
    llm: Optional[BaseChatModel] = None,
    llm_provider: str = "openai",
    llm_model: Optional[str] = None,
) -> ReActNavigator:
    """Factory function to create a ReAct navigator.
    
    Args:
        env: GridWorld environment
        llm: Optional pre-configured LLM instance
        llm_provider: Provider name if no LLM given ('openai', 'deepseek', 'kimi')
        llm_model: Model name override (defaults to provider's default)
    """
    return ReActNavigator(env, llm=llm, llm_provider=llm_provider, llm_model=llm_model)


def create_heuristic_navigator(
    env: GridWorld,
    use_bfs_hint: bool = True,
    use_memory: bool = True,
) -> HeuristicNavigator:
    """Factory function to create a heuristic navigator."""
    return HeuristicNavigator(env, use_bfs_hint=use_bfs_hint, use_memory=use_memory)
