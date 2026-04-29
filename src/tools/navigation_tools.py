"""
LangChain tools for gridworld navigation.
"""

from typing import List, Optional
from langchain_core.tools import tool
from langchain_core.tools.base import BaseTool

from ..env.gridworld import GridWorld


def create_navigation_tools(env: GridWorld) -> List[BaseTool]:
    """Create LangChain tools bound to a GridWorld instance."""
    
    @tool
    def sense_surroundings() -> str:
        """Sense the surroundings of the agent and return what's nearby.
        
        Returns a description of the 3x3 area around the agent including
        obstacles, empty spaces, and the goal position.
        """
        obs = env._get_obs()
        ar, ac = obs["agent_pos"]
        gr, gc = obs["goal_pos"]
        
        # Build 3x3 grid description
        rows = []
        for dr in [-1, 0, 1]:
            row = []
            for dc in [-1, 0, 1]:
                nr, nc = ar + dr, ac + dc
                if dr == 0 and dc == 0:
                    row.append("A")  # Agent
                elif (nr, nc) == (gr, gc):
                    row.append("G")  # Goal
                elif 0 <= nr < env.size and 0 <= nc < env.size:
                    if env.grid[nr, nc] == 1:  # OBSTACLE
                        row.append("#")
                    else:
                        row.append(".")
                else:
                    row.append("X")  # Out of bounds
            rows.append(" ".join(row))
        
        grid_view = "\n".join(rows)
        
        # Direction to goal
        dr = gr - ar
        dc = gc - ac
        direction_hints = []
        if dr < 0:
            direction_hints.append("UP")
        elif dr > 0:
            direction_hints.append("DOWN")
        if dc < 0:
            direction_hints.append("LEFT")
        elif dc > 0:
            direction_hints.append("RIGHT")
        
        return (
            f"Agent at ({ar}, {ac}), Goal at ({gr}, {gc})\n"
            f"Manhattan distance: {obs['distance_to_goal']}\n"
            f"Goal is to the: {', '.join(direction_hints) if direction_hints else 'HERE'}\n"
            f"Surroundings (3x3, A=agent, G=goal, #=obstacle, .=empty, X=out of bounds):\n"
            f"{grid_view}"
        )
    
    @tool
    def move(direction: str) -> str:
        """Move the agent in a direction.
        
        Args:
            direction: One of 'up', 'down', 'left', 'right'
        
        Returns:
            Result of the move including new position and any reward.
        """
        obs, reward, done, info = env.step(direction)
        ar, ac = obs["agent_pos"]
        gr, gc = obs["goal_pos"]
        
        result = f"Moved {direction}. Now at ({ar}, {ac}). Goal at ({gr}, {gc}). "
        result += f"Distance: {obs['distance_to_goal']}. Reward: {reward:.2f}."
        
        if done:
            if info.get("reason") == "reached_goal":
                result += " \n🎉 SUCCESS! Reached the goal!"
            else:
                result += f" \nEpisode ended: {info.get('reason', 'unknown')}"
        
        if "error" in info:
            result += f" \nError: {info['error']}"
        elif "reason" in info and info["reason"] in ("out_of_bounds", "hit_obstacle"):
            result += f" \nBlocked! {info['reason']}"
        
        return result
    
    @tool
    def get_position() -> str:
        """Get the current position of the agent and the goal."""
        obs = env._get_obs()
        ar, ac = obs["agent_pos"]
        gr, gc = obs["goal_pos"]
        return (
            f"Agent: ({ar}, {ac}), Goal: ({gr}, {gc}), "
            f"Distance: {obs['distance_to_goal']}"
        )
    
    @tool
    def get_path_hint() -> str:
        """Get a hint about the shortest path to the goal.
        
        This uses BFS to find the optimal next step.
        """
        path = env.get_shortest_path()
        if path is None:
            return "No path found to goal!"
        if len(path) <= 1:
            return "Already at goal!"
        
        current = env.agent_pos
        next_step = path[1]
        
        dr = next_step[0] - current[0]
        dc = next_step[1] - current[1]
        
        direction = ""
        if dr == -1:
            direction = "UP"
        elif dr == 1:
            direction = "DOWN"
        elif dc == -1:
            direction = "LEFT"
        elif dc == 1:
            direction = "RIGHT"
        
        return (
            f"Hint: Move {direction} to ({next_step[0]}, {next_step[1]}). "
            f"Shortest path length: {len(path) - 1} steps."
        )
    
    return [sense_surroundings, move, get_position, get_path_hint]
