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


class NavigationAgent:
    """Base class for navigation agents."""
    
    def __init__(self, env: GridWorld):
        self.env = env
        self.tools = create_navigation_tools(env)
        self.history: List[Dict[str, Any]] = []
    
    def act(self, observation: Optional[Dict] = None) -> str:
        """Take an action based on observation. Returns action name."""
        raise NotImplementedError
    
    def reset(self):
        """Reset agent state."""
        self.history.clear()


class HeuristicNavigator(NavigationAgent):
    """
    A heuristic-based navigator that doesn't require an LLM.
    Uses simple wall-following / goal-seeking behavior.
    """
    
    def __init__(self, env: GridWorld, use_bfs_hint: bool = True):
        super().__init__(env)
        self.use_bfs_hint = use_bfs_hint
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
            """Score an action for local navigation."""
            dr, dc = self.env.DIRECTIONS[action]
            nr, nc = ar + dr, ac + dc
            score = 0.0
            
            # Base: prefer moves toward goal direction even when not visible
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
                # Visible goal: minimize Manhattan distance
                score += abs(nr - gr) + abs(nc - gc)
            else:
                # Goal not visible: prefer exploring new areas
                if not self.env.visited_mask[nr, nc]:
                    score -= 5.0  # Reward unvisited cells
                else:
                    score += 1.0  # Slight penalty for visited cells
            
            # Strong penalty for revisiting very recent positions (avoid 2-step loops)
            if (nr, nc) in self.last_positions[-3:]:
                score += 20.0
            elif (nr, nc) in self.last_positions:
                score += 5.0
            
            return score
        
        # Score each action and pick the best
        best_action = None
        best_score = float('inf')
        
        for action in valid:
            score = _score_action(action)
            if score < best_score:
                best_score = score
                best_action = action
        
        if best_action is None and valid:
            best_action = valid[0]
        
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


class ReActNavigator(NavigationAgent):
    """
    A ReAct navigator powered by an LLM via LangGraph.
    """
    
    def __init__(
        self,
        env: GridWorld,
        llm: Optional[BaseChatModel] = None,
        max_iterations: int = 50,
    ):
        super().__init__(env)
        self.max_iterations = max_iterations
        
        if llm is None:
            api_key = os.environ.get("OPENAI_API_KEY")
            if api_key:
                llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
            else:
                raise ValueError(
                    "No LLM provided and OPENAI_API_KEY not set. "
                    "Please provide an LLM or set the API key."
                )
        
        self.llm = llm
        self.agent = create_react_agent(
            model=llm,
            tools=self.tools,
        )
        self.messages: List[Any] = []
    
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
        """Build the navigation prompt for the LLM."""
        obs = self.env._get_obs()
        ar, ac = obs["agent_pos"]
        gr, gc = obs["goal_pos"]
        
        valid = self.env.get_valid_actions()
        
        prompt = f"""You are a navigation agent in a {self.env.size}x{self.env.size} grid world.
Your current position: ({ar}, {ac})
Goal position: ({gr}, {gc})
Distance to goal: {obs['distance_to_goal']}
Valid moves from here: {valid}

You must navigate to the goal (G) while avoiding obstacles (#).
Use the available tools to sense surroundings and move.

Strategy:
1. First, use sense_surroundings to understand the environment
2. Then use move() with one of: up, down, left, right
3. You can use get_path_hint() if you're stuck

Think step by step. Make ONE move at a time.
"""
        return prompt
    
    def reset(self):
        super().reset()
        self.messages.clear()


def create_react_navigator(
    env: GridWorld,
    llm: Optional[BaseChatModel] = None,
) -> ReActNavigator:
    """Factory function to create a ReAct navigator."""
    return ReActNavigator(env, llm=llm)


def create_heuristic_navigator(
    env: GridWorld,
    use_bfs_hint: bool = True,
) -> HeuristicNavigator:
    """Factory function to create a heuristic navigator."""
    return HeuristicNavigator(env, use_bfs_hint=use_bfs_hint)
