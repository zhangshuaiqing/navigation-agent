"""
GridWorld environment for navigation tasks.
"""

import random
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Optional, Tuple, List
import numpy as np


class CellType(IntEnum):
    EMPTY = 0
    OBSTACLE = 1
    AGENT = 2
    GOAL = 3
    PATH = 4
    DYNAMIC_OBSTACLE = 5


@dataclass
class DynamicObstacle:
    """A moving obstacle in the grid world."""
    pos: Tuple[int, int]
    direction: str = "right"  # up, down, left, right
    speed: int = 1  # Move every N agent steps
    move_prob: float = 0.5  # Probability of moving when speed triggers
    boundary_mode: str = "bounce"  # bounce, wrap, random


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
        observation_mode: str = "full",
        view_range: int = 1,
        num_dynamic_obstacles: int = 0,
        dynamic_obstacle_speed: int = 1,
    ):
        self.size = size
        self.obstacle_ratio = obstacle_ratio
        self.random_start_goal = random_start_goal
        self.observation_mode = observation_mode
        self.view_range = max(1, view_range)
        self.num_dynamic_obstacles = num_dynamic_obstacles
        self.dynamic_obstacle_speed = dynamic_obstacle_speed
        self.rng = random.Random(seed)
        self.np_rng = np.random.default_rng(seed)
        
        self.grid: np.ndarray = np.zeros((size, size), dtype=int)
        self.agent_pos: Tuple[int, int] = (0, 0)
        self.goal_pos: Tuple[int, int] = (size - 1, size - 1)
        self.step_count: int = 0
        self.max_steps: int = size * size * 2
        self.done: bool = False
        self.visited_mask: np.ndarray = np.zeros((size, size), dtype=bool)
        self.dynamic_obstacles: List[DynamicObstacle] = []
        
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
        
        # Initialize visited mask for fog_of_war mode
        self.visited_mask.fill(False)
        self._update_visited_mask()
        
        # Initialize dynamic obstacles
        self._init_dynamic_obstacles()
    
    def _init_dynamic_obstacles(self):
        """Initialize dynamic obstacles on empty cells."""
        self.dynamic_obstacles.clear()
        if self.num_dynamic_obstacles <= 0:
            return
        
        directions = ["up", "down", "left", "right"]
        boundary_modes = ["bounce", "bounce", "bounce", "random"]  # Mostly bounce
        
        # Collect empty cells (not agent, not goal, not static obstacle)
        empty_cells = []
        for r in range(self.size):
            for c in range(self.size):
                if self.grid[r, c] == CellType.EMPTY:
                    empty_cells.append((r, c))
        
        self.rng.shuffle(empty_cells)
        
        for i in range(min(self.num_dynamic_obstacles, len(empty_cells))):
            pos = empty_cells[i]
            dyn_obs = DynamicObstacle(
                pos=pos,
                direction=self.rng.choice(directions),
                speed=self.dynamic_obstacle_speed,
                move_prob=self.rng.uniform(0.3, 0.7),
                boundary_mode=self.rng.choice(boundary_modes),
            )
            self.dynamic_obstacles.append(dyn_obs)
            self.grid[pos] = CellType.DYNAMIC_OBSTACLE
    
    def _pick_start_goal(self):
        """Randomly pick start and goal positions from empty cells."""
        all_cells = [(r, c) for r in range(self.size) for c in range(self.size)]
        self.rng.shuffle(all_cells)
        self.agent_pos = all_cells[0]
        self.goal_pos = all_cells[1]
    
    def _update_visited_mask(self):
        """Mark cells within view_range as visited."""
        ar, ac = self.agent_pos
        vr = self.view_range
        for r in range(max(0, ar - vr), min(self.size, ar + vr + 1)):
            for c in range(max(0, ac - vr), min(self.size, ac + vr + 1)):
                self.visited_mask[r, c] = True
    
    def _update_dynamic_obstacles(self):
        """Move dynamic obstacles after agent step."""
        if not self.dynamic_obstacles:
            return
        
        # Only move every N agent steps based on speed
        if self.step_count % self.dynamic_obstacle_speed != 0:
            return
        
        all_dirs = ["up", "down", "left", "right"]
        opposite = {"up": "down", "down": "up", "left": "right", "right": "left"}
        dir_deltas = {"up": (-1, 0), "down": (1, 0), "left": (0, -1), "right": (0, 1)}
        
        for dyn in self.dynamic_obstacles:
            if self.rng.random() > dyn.move_prob:
                continue
            
            r, c = dyn.pos
            dr, dc = dir_deltas[dyn.direction]
            nr, nc = r + dr, c + dc
            
            # Check boundaries
            out_of_bounds = not (0 <= nr < self.size and 0 <= nc < self.size)
            
            if out_of_bounds:
                if dyn.boundary_mode == "bounce":
                    dyn.direction = opposite[dyn.direction]
                elif dyn.boundary_mode == "wrap":
                    nr = (nr + self.size) % self.size
                    nc = (nc + self.size) % self.size
                elif dyn.boundary_mode == "random":
                    dyn.direction = self.rng.choice(all_dirs)
                    dr, dc = dir_deltas[dyn.direction]
                    nr, nc = r + dr, c + dc
                    if not (0 <= nr < self.size and 0 <= nc < self.size):
                        continue
                else:
                    continue
            
            # Check if target cell is valid (not static obstacle, not agent, not goal, not another dynamic)
            target_cell = self.grid[nr, nc]
            if target_cell in (CellType.OBSTACLE, CellType.AGENT, CellType.GOAL, CellType.DYNAMIC_OBSTACLE):
                # Blocked: try to bounce or pick random direction
                if dyn.boundary_mode == "bounce":
                    dyn.direction = opposite[dyn.direction]
                else:
                    # Pick a random valid direction
                    valid_dirs = []
                    for d in all_dirs:
                        ddr, ddc = dir_deltas[d]
                        nnr, nnc = r + ddr, c + ddc
                        if 0 <= nnr < self.size and 0 <= nnc < self.size:
                            if self.grid[nnr, nnc] not in (CellType.OBSTACLE, CellType.AGENT, CellType.GOAL, CellType.DYNAMIC_OBSTACLE):
                                valid_dirs.append(d)
                    if valid_dirs:
                        dyn.direction = self.rng.choice(valid_dirs)
                        dr, dc = dir_deltas[dyn.direction]
                        nr, nc = r + dr, c + dc
                    else:
                        continue
                # Recheck after direction change
                target_cell = self.grid[nr, nc]
                if target_cell in (CellType.OBSTACLE, CellType.AGENT, CellType.GOAL, CellType.DYNAMIC_OBSTACLE):
                    continue
            
            # Move dynamic obstacle
            self.grid[r, c] = CellType.EMPTY
            dyn.pos = (nr, nc)
            self.grid[nr, nc] = CellType.DYNAMIC_OBSTACLE
    
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
            # Just reset positions and dynamic obstacles
            self.grid[self.grid == CellType.AGENT] = CellType.EMPTY
            self.grid[self.grid == CellType.DYNAMIC_OBSTACLE] = CellType.EMPTY
            self.grid[self.agent_pos] = CellType.AGENT
            self.grid[self.goal_pos] = CellType.GOAL
            self._init_dynamic_obstacles()
        
        # Reset visited mask
        self.visited_mask.fill(False)
        self._update_visited_mask()
        
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
        # Check static obstacle
        elif self.grid[new_r, new_c] == CellType.OBSTACLE:
            reward = -0.5
            info = {"reason": "hit_obstacle"}
        # Check dynamic obstacle collision
        elif self.grid[new_r, new_c] == CellType.DYNAMIC_OBSTACLE:
            reward = -1.0
            info = {"reason": "hit_dynamic_obstacle", "agent_bounced": True}
            # Agent does not move, stays in place
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
        
        # Update dynamic obstacles after agent moves
        self._update_dynamic_obstacles()
        
        self.step_count += 1
        if self.step_count >= self.max_steps:
            self.done = True
            info["reason"] = info.get("reason", "") + "_max_steps"
        
        # Update visited mask after agent moves
        self._update_visited_mask()
        
        return self._get_obs(), reward, self.done, info
    
    def _get_obs(self) -> dict:
        """Get current observation based on observation_mode."""
        ar, ac = self.agent_pos
        gr, gc = self.goal_pos
        vr = self.view_range
        
        # Build surroundings based on view_range
        surroundings = []
        for dr in range(-vr, vr + 1):
            for dc in range(-vr, vr + 1):
                nr, nc = ar + dr, ac + dc
                if 0 <= nr < self.size and 0 <= nc < self.size:
                    if self.observation_mode == "fog_of_war" and not self.visited_mask[nr, nc]:
                        surroundings.append("?")
                    elif (nr, nc) == self.agent_pos:
                        surroundings.append("A")
                    elif (nr, nc) == self.goal_pos:
                        surroundings.append("G")
                    elif self.grid[nr, nc] == CellType.OBSTACLE:
                        surroundings.append("#")
                    else:
                        surroundings.append(".")
                else:
                    surroundings.append("X")
        
        # Distance to goal (always available in full, hidden in local/fog)
        if self.observation_mode == "full":
            distance = abs(ar - gr) + abs(ac - gc)
            goal_visible = True
        else:
            # In local/fog modes, only show distance if goal is within view
            goal_visible = abs(ar - gr) <= vr and abs(ac - gc) <= vr
            distance = abs(ar - gr) + abs(ac - gc) if goal_visible else None
        
        # Goal direction hint (available in all modes, but not exact position)
        goal_direction = None
        if gr is not None and gc is not None:
            dr = gr - ar
            dc = gc - ac
            dirs = []
            if dr < 0:
                dirs.append("N")
            elif dr > 0:
                dirs.append("S")
            if dc < 0:
                dirs.append("W")
            elif dc > 0:
                dirs.append("E")
            goal_direction = "".join(dirs) if dirs else "HERE"
        
        return {
            "agent_pos": self.agent_pos,
            "goal_pos": self.goal_pos if self.observation_mode == "full" else None,
            "goal_visible": goal_visible,
            "goal_direction": goal_direction,
            "grid_size": self.size,
            "surroundings": surroundings,
            "distance_to_goal": distance,
            "step_count": self.step_count,
            "max_steps": self.max_steps,
            "observation_mode": self.observation_mode,
            "view_range": self.view_range,
        }
    
    def render(self, mark_path: Optional[List[Tuple[int, int]]] = None, show_fog: bool = False) -> str:
        """Render the grid as a string."""
        lines = []
        for r in range(self.size):
            row_str = ""
            for c in range(self.size):
                # Fog of war: hide unvisited cells
                if show_fog and self.observation_mode == "fog_of_war" and not self.visited_mask[r, c]:
                    row_str += " ? "
                elif (r, c) == self.agent_pos:
                    row_str += " A "
                elif (r, c) == self.goal_pos:
                    row_str += " G "
                elif mark_path and (r, c) in mark_path:
                    row_str += " * "
                elif self.grid[r, c] == CellType.OBSTACLE:
                    row_str += "###"
                elif self.grid[r, c] == CellType.DYNAMIC_OBSTACLE:
                    row_str += " D "
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
                and self.grid[nr, nc] not in (CellType.OBSTACLE, CellType.DYNAMIC_OBSTACLE)
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
