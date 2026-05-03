"""
GridWorld environment for navigation tasks.
"""

import random
from enum import IntEnum
from typing import Optional, Tuple, List
import numpy as np


class CellType(IntEnum):
    EMPTY = 0
    OBSTACLE = 1
    AGENT = 2
    GOAL = 3
    PATH = 4


class GridWorld:
    """
    A simple grid world environment.
    
    The grid uses (row, col) coordinates where (0,0) is top-left.
    Actions: 0=up, 1=down, 2=left, 3=right
    """
    
    ACTIONS = ["up", "down", "left", "right"]
    DIRECTIONS = {
        "up": (-1, 0),
        "down": (1, 0),
        "left": (0, -1),
        "right": (0, 1),
    }
    
    def __init__(
        self,
        size: int = 8,
        obstacle_ratio: float = 0.2,
        seed: Optional[int] = None,
        random_start_goal: bool = False,
    ):
        self.size = size
        self.obstacle_ratio = obstacle_ratio
        self.random_start_goal = random_start_goal
        self.rng = random.Random(seed)
        self.np_rng = np.random.default_rng(seed)
        
        self.grid: np.ndarray = np.zeros((size, size), dtype=int)
        self.agent_pos: Tuple[int, int] = (0, 0)
        self.goal_pos: Tuple[int, int] = (size - 1, size - 1)
        self.step_count: int = 0
        self.max_steps: int = size * size * 2
        self.done: bool = False
        
        self._generate_map()
    
    def _generate_map(self):
        """Generate a random map with obstacles."""
        self.grid.fill(CellType.EMPTY)
        
        # Pick start/goal positions if random mode
        if self.random_start_goal:
            self._pick_start_goal()
        else:
            self.agent_pos = (0, 0)
            self.goal_pos = (self.size - 1, self.size - 1)
        
        # Place obstacles
        num_obstacles = int(self.size * self.size * self.obstacle_ratio)
        flat_indices = self.np_rng.choice(
            self.size * self.size, size=num_obstacles, replace=False
        )
        for idx in flat_indices:
            r, c = divmod(idx, self.size)
            # Don't block start or goal
            if (r, c) != self.agent_pos and (r, c) != self.goal_pos:
                self.grid[r, c] = CellType.OBSTACLE
        
        # Ensure path exists (simple check: BFS from start to goal)
        if not self._path_exists(self.agent_pos, self.goal_pos):
            self._clear_path_to_goal()
        
        self.grid[self.agent_pos] = CellType.AGENT
        self.grid[self.goal_pos] = CellType.GOAL
    
    def _pick_start_goal(self):
        """Randomly pick start and goal positions from empty cells."""
        all_cells = [(r, c) for r in range(self.size) for c in range(self.size)]
        self.rng.shuffle(all_cells)
        self.agent_pos = all_cells[0]
        self.goal_pos = all_cells[1]
    
    def _path_exists(
        self, start: Tuple[int, int], goal: Tuple[int, int]
    ) -> bool:
        """Check if a path exists using BFS."""
        from collections import deque
        
        visited = set([start])
        queue = deque([start])
        
        while queue:
            r, c = queue.popleft()
            if (r, c) == goal:
                return True
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = r + dr, c + dc
                if (
                    0 <= nr < self.size
                    and 0 <= nc < self.size
                    and (nr, nc) not in visited
                    and self.grid[nr, nc] != CellType.OBSTACLE
                ):
                    visited.add((nr, nc))
                    queue.append((nr, nc))
        return False
    
    def _clear_path_to_goal(self):
        """Clear obstacles along a random path to ensure solvability."""
        r, c = self.agent_pos
        gr, gc = self.goal_pos
        
        # Create an L-shaped or random path
        while (r, c) != (gr, gc):
            if self.rng.random() < 0.5 and r != gr:
                r += 1 if gr > r else -1
            elif c != gc:
                c += 1 if gc > c else -1
            else:
                r += 1 if gr > r else -1
            if self.grid[r, c] == CellType.OBSTACLE:
                self.grid[r, c] = CellType.EMPTY
    
    def reset(
        self,
        agent_pos: Optional[Tuple[int, int]] = None,
        goal_pos: Optional[Tuple[int, int]] = None,
        new_map: bool = False,
        random_start_goal: Optional[bool] = None,
    ) -> dict:
        """Reset the environment."""
        self.step_count = 0
        self.done = False
        
        # Update random_start_goal setting if provided
        if random_start_goal is not None:
            self.random_start_goal = random_start_goal
        
        if agent_pos is not None:
            self.agent_pos = agent_pos
        elif self.random_start_goal and new_map:
            self._pick_start_goal()
        elif self.random_start_goal:
            # Keep current positions but ensure they're set
            if self.agent_pos == self.goal_pos:
                self._pick_start_goal()
        else:
            self.agent_pos = (0, 0)
        
        if goal_pos is not None:
            self.goal_pos = goal_pos
        elif not self.random_start_goal:
            self.goal_pos = (self.size - 1, self.size - 1)
        
        if new_map:
            self._generate_map()
        else:
            # Just reset positions
            self.grid[self.grid == CellType.AGENT] = CellType.EMPTY
            self.grid[self.agent_pos] = CellType.AGENT
            self.grid[self.goal_pos] = CellType.GOAL
        
        return self._get_obs()
    
    def step(self, action: str) -> Tuple[dict, float, bool, dict]:
        """
        Execute an action.
        
        Returns: (observation, reward, done, info)
        """
        if self.done:
            return self._get_obs(), 0.0, True, {"reason": "already_done"}
        
        action = action.lower().strip()
        if action not in self.ACTIONS:
            return self._get_obs(), -0.1, False, {"error": f"Invalid action: {action}"}
        
        dr, dc = self.DIRECTIONS[action]
        new_r, new_c = self.agent_pos[0] + dr, self.agent_pos[1] + dc
        
        # Check bounds
        if not (0 <= new_r < self.size and 0 <= new_c < self.size):
            reward = -0.5
            info = {"reason": "out_of_bounds", "attempted": (new_r, new_c)}
        # Check obstacle
        elif self.grid[new_r, new_c] == CellType.OBSTACLE:
            reward = -0.5
            info = {"reason": "hit_obstacle"}
        else:
            # Move agent
            old_r, old_c = self.agent_pos
            self.grid[old_r, old_c] = CellType.EMPTY
            self.agent_pos = (new_r, new_c)
            self.grid[new_r, new_c] = CellType.AGENT
            
            # Check goal
            if self.agent_pos == self.goal_pos:
                reward = 10.0
                self.done = True
                info = {"reason": "reached_goal"}
            else:
                # Distance-based reward
                old_dist = abs(old_r - self.goal_pos[0]) + abs(old_c - self.goal_pos[1])
                new_dist = abs(new_r - self.goal_pos[0]) + abs(new_c - self.goal_pos[1])
                reward = 0.1 if new_dist < old_dist else -0.1
                info = {"reason": "moved", "distance_to_goal": new_dist}
        
        self.step_count += 1
        if self.step_count >= self.max_steps:
            self.done = True
            info["reason"] = info.get("reason", "") + "_max_steps"
        
        return self._get_obs(), reward, self.done, info
    
    def _get_obs(self) -> dict:
        """Get current observation."""
        ar, ac = self.agent_pos
        gr, gc = self.goal_pos
        
        # Get surrounding cells (3x3 neighborhood)
        surroundings = []
        for dr in [-1, 0, 1]:
            for dc in [-1, 0, 1]:
                nr, nc = ar + dr, ac + dc
                if 0 <= nr < self.size and 0 <= nc < self.size:
                    cell = self.grid[nr, nc]
                    if (nr, nc) == self.agent_pos:
                        surroundings.append("A")
                    elif (nr, nc) == self.goal_pos:
                        surroundings.append("G")
                    elif cell == CellType.OBSTACLE:
                        surroundings.append("#")
                    else:
                        surroundings.append(".")
                else:
                    surroundings.append("X")
        
        return {
            "agent_pos": self.agent_pos,
            "goal_pos": self.goal_pos,
            "grid_size": self.size,
            "surroundings": surroundings,
            "distance_to_goal": abs(ar - gr) + abs(ac - gc),
            "step_count": self.step_count,
            "max_steps": self.max_steps,
        }
    
    def render(self, mark_path: Optional[List[Tuple[int, int]]] = None) -> str:
        """Render the grid as a string."""
        lines = []
        for r in range(self.size):
            row_str = ""
            for c in range(self.size):
                if (r, c) == self.agent_pos:
                    row_str += " A "
                elif (r, c) == self.goal_pos:
                    row_str += " G "
                elif mark_path and (r, c) in mark_path:
                    row_str += " * "
                elif self.grid[r, c] == CellType.OBSTACLE:
                    row_str += "###"
                else:
                    row_str += " . "
            lines.append(row_str)
        return "\n".join(lines)
    
    def get_valid_actions(self) -> List[str]:
        """Get list of valid actions from current position."""
        valid = []
        ar, ac = self.agent_pos
        for action, (dr, dc) in self.DIRECTIONS.items():
            nr, nc = ar + dr, ac + dc
            if (
                0 <= nr < self.size
                and 0 <= nc < self.size
                and self.grid[nr, nc] != CellType.OBSTACLE
            ):
                valid.append(action)
        return valid
    
    def get_shortest_path(self) -> Optional[List[Tuple[int, int]]]:
        """Get shortest path from agent to goal using BFS."""
        from collections import deque
        
        start = self.agent_pos
        goal = self.goal_pos
        
        if start == goal:
            return [start]
        
        queue = deque([(start, [start])])
        visited = {start}
        
        while queue:
            (r, c), path = queue.popleft()
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = r + dr, c + dc
                if (
                    0 <= nr < self.size
                    and 0 <= nc < self.size
                    and (nr, nc) not in visited
                    and self.grid[nr, nc] != CellType.OBSTACLE
                ):
                    if (nr, nc) == goal:
                        return path + [(nr, nc)]
                    visited.add((nr, nc))
                    queue.append(((nr, nc), path + [(nr, nc)]))
        
        return None
