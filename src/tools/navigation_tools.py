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
        
        Returns a description of the area around the agent including
        obstacles, empty spaces, and the goal position.
        The visible range depends on the environment's view_range setting.
        """
        obs = env._get_obs()
        ar, ac = obs["agent_pos"]
        gr, gc = obs["goal_pos"] or (None, None)
        vr = obs["view_range"]
        
        # Build grid description based on view_range
        rows = []
        for dr in range(-vr, vr + 1):
            row = []
            for dc in range(-vr, vr + 1):
                nr, nc = ar + dr, ac + dc
                if dr == 0 and dc == 0:
                    row.append("A")  # Agent
                elif gr is not None and (nr, nc) == (gr, gc):
                    row.append("G")  # Goal
                elif 0 <= nr < env.size and 0 <= nc < env.size:
                    cell = env.grid[nr, nc]
                    if env.observation_mode == "fog_of_war" and not env.visited_mask[nr, nc]:
                        row.append("?")
                    elif cell == 1:  # OBSTACLE
                        row.append("#")
                    else:
                        row.append(".")
                else:
                    row.append("X")  # Out of bounds
            rows.append(" ".join(row))
        
        grid_view = "\n".join(rows)
        
        # Direction to goal
        if gr is not None and gc is not None:
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
            dir_str = f"Goal is to the: {', '.join(direction_hints) if direction_hints else 'HERE'}"
        else:
            dir_str = "Goal direction: UNKNOWN (not in view)"
        
        dist_str = (
            f"Manhattan distance: {obs['distance_to_goal']}"
            if obs['distance_to_goal'] is not None
            else "Distance to goal: UNKNOWN (not in view)"
        )
        
        mode_note = ""
        if env.observation_mode == "local":
            mode_note = "\n[LOCAL MODE: Only local surroundings visible, no global path hints]"
        elif env.observation_mode == "fog_of_war":
            mode_note = "\n[FOG OF WAR: Unvisited cells shown as '?']"
        
        return (
            f"Agent at ({ar}, {ac})"
            + (f", Goal at ({gr}, {gc})" if gr is not None else "")
            + f"\n{dist_str}\n"
            + f"{dir_str}\n"
            + f"Surroundings ({2*vr+1}x{2*vr+1}, A=agent, G=goal, #=obstacle, .=empty, ?=unseen, X=out of bounds):\n"
            + f"{grid_view}"
            + mode_note
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
        gr, gc = obs["goal_pos"] or (None, None)
        
        result = f"Moved {direction}. Now at ({ar}, {ac})."
        if gr is not None:
            result += f" Goal at ({gr}, {gc})."
        
        if obs["distance_to_goal"] is not None:
            result += f" Distance: {obs['distance_to_goal']}."
        result += f" Reward: {reward:.2f}."
        
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
        gr, gc = obs["goal_pos"] or (None, None)
        
        if gr is None:
            # Local/fog mode: goal not visible, give relative direction only
            return (
                f"Agent: ({ar}, {ac}). "
                f"Goal is not currently visible in your view range ({obs['view_range']}). "
                f"Explore to find it!"
            )
        
        return (
            f"Agent: ({ar}, {ac}), Goal: ({gr}, {gc}), "
            f"Distance: {obs['distance_to_goal']}"
        )
    
    @tool
    def get_path_hint() -> str:
        """Get a hint about the shortest path to the goal.
        
        This uses BFS to find the optimal next step.
        In local/fog observation modes, this tool is disabled.
        """
        if env.observation_mode != "full":
            return (
                f"Path hints are unavailable in '{env.observation_mode}' mode. "
                f"You must navigate using only local surroundings."
            )
        
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
