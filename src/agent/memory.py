"""
Agent memory module for navigation agents.

Provides memory of visited cells, obstacles, trajectory, and goal positions.
Supports dead-end detection, fork detection, and backtracking.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple


@dataclass
class AgentMemory:
    """
    Memory of an agent's exploration history.
    
    Attributes:
        visited: All positions the agent has visited.
        trajectory: Ordered list of positions visited (full path).
        walls: All obstacle positions the agent has observed.
        goals_found: All goal positions the agent has seen.
        forks: Indices into trajectory pointing to fork points
               (positions with >= 2 unexplored adjacent cells).
        last_actions: Recent action history for loop detection.
    """
    visited: Set[Tuple[int, int]] = field(default_factory=set)
    trajectory: List[Tuple[int, int]] = field(default_factory=list)
    walls: Set[Tuple[int, int]] = field(default_factory=set)
    goals_found: List[Tuple[int, int]] = field(default_factory=list)
    forks: List[int] = field(default_factory=list)
    last_actions: List[str] = field(default_factory=list)
    
    max_trajectory: int = 200
    max_forks: int = 50
    
    def update(self, pos: Tuple[int, int], obs: dict, actions: List[str], grid_size: int):
        """Update memory based on current observation and position."""
        self.visited.add(pos)
        
        # Track trajectory (limit size)
        if not self.trajectory or self.trajectory[-1] != pos:
            self.trajectory.append(pos)
        if len(self.trajectory) > self.max_trajectory:
            self.trajectory.pop(0)
        
        # Track goal positions
        goal_pos = obs.get("goal_pos")
        if goal_pos is not None and goal_pos not in self.goals_found:
            self.goals_found.append(goal_pos)
        
        # Detect obstacles in surroundings
        vr = obs.get("view_range", 1)
        surroundings = obs.get("surroundings", [])
        if surroundings:
            side = 2 * vr + 1
            for dr in range(-vr, vr + 1):
                for dc in range(-vr, vr + 1):
                    idx = (dr + vr) * side + (dc + vr)
                    if idx < len(surroundings):
                        nr, nc = pos[0] + dr, pos[1] + dc
                        if 0 <= nr < grid_size and 0 <= nc < grid_size:
                            char = surroundings[idx]
                            if char == "#":
                                self.walls.add((nr, nc))
        
        # Track last actions
        if actions:
            self.last_actions = actions[-10:]  # Keep last 10
    
    def detect_fork(self, pos: Tuple[int, int], grid_size: int,
                    valid_actions: List[str]) -> bool:
        """Check if current position is a fork (>=2 unexplored exits)."""
        unexplored = self._count_unexplored_adjacent(pos, grid_size, valid_actions)
        return unexplored >= 2
    
    def _count_unexplored_adjacent(self, pos: Tuple[int, int], grid_size: int,
                                   valid_actions: List[str]) -> int:
        """Count adjacent cells that are both valid and unvisited."""
        directions = {
            "up": (-1, 0), "down": (1, 0),
            "left": (0, -1), "right": (0, 1),
        }
        count = 0
        for action in valid_actions:
            dr, dc = directions.get(action, (0, 0))
            nr, nc = pos[0] + dr, pos[1] + dc
            if 0 <= nr < grid_size and 0 <= nc < grid_size:
                if (nr, nc) not in self.visited and (nr, nc) not in self.walls:
                    count += 1
        return count
    
    def record_fork(self, pos: Tuple[int, int], grid_size: int,
                    valid_actions: List[str]):
        """Record current position as a fork point."""
        if len(self.forks) >= self.max_forks:
            self.forks.pop(0)
        # Find current index in trajectory
        try:
            idx = len(self.trajectory) - 1
            if idx >= 0 and idx not in self.forks:
                self.forks.append(idx)
        except (ValueError, IndexError):
            pass
    
    def is_dead_end(self, pos: Tuple[int, int], grid_size: int,
                    valid_actions: List[str]) -> bool:
        """Check if current position is a dead end (no unexplored exits)."""
        if not valid_actions:
            return True
        unexplored = self._count_unexplored_adjacent(pos, grid_size, valid_actions)
        # It's a dead end if all valid moves lead to visited/wall cells
        return unexplored == 0
    
    def find_unvisited_direction(self, pos: Tuple[int, int], grid_size: int,
                                 valid_actions: List[str]) -> Optional[str]:
        """Find a direction that leads to an unvisited cell."""
        directions = {
            "up": (-1, 0), "down": (1, 0),
            "left": (0, -1), "right": (0, 1),
        }
        for action in valid_actions:
            dr, dc = directions.get(action, (0, 0))
            nr, nc = pos[0] + dr, pos[1] + dc
            if 0 <= nr < grid_size and 0 <= nc < grid_size:
                if (nr, nc) not in self.visited and (nr, nc) not in self.walls:
                    return action
        return None
    
    def backtrack(self, pos: Tuple[int, int]) -> Optional[str]:
        """
        Find the next action to backtrack toward the nearest fork point.
        
        Returns the direction to move to retrace the trajectory, or None
        if no backtracking path exists.
        """
        if len(self.trajectory) < 2:
            return None
        
        # Find the nearest fork point before current position
        current_idx = len(self.trajectory) - 1
        target_idx = None
        
        for idx in reversed(self.forks):
            if idx < current_idx:
                target_idx = idx
                break
        
        if target_idx is None:
            # No fork: backtrack to the previous position
            target_idx = max(0, current_idx - 1)
        
        # We need to move from pos to trajectory[target_idx]
        # But we can only move one step - so move toward trajectory[current_idx - 1]
        backtrack_idx = max(0, current_idx - 1)
        target = self.trajectory[backtrack_idx]
        
        dr = target[0] - pos[0]
        dc = target[1] - pos[1]
        
        if dr == -1:
            return "up"
        elif dr == 1:
            return "down"
        elif dc == -1:
            return "left"
        elif dc == 1:
            return "right"
        
        return None
    
    def get_summary(self) -> str:
        """Get a text summary of the memory for prompt injection."""
        parts = []
        parts.append(f"Visited {len(self.visited)} cells")
        parts.append(f"Detected {len(self.walls)} obstacles")
        parts.append(f"Found {len(self.goals_found)} goal(s)")
        if self.forks:
            parts.append(f"{len(self.forks)} fork point(s) recorded")
        return " | ".join(parts)
    
    def reset(self):
        """Clear all memory."""
        self.visited.clear()
        self.trajectory.clear()
        self.walls.clear()
        self.goals_found.clear()
        self.forks.clear()
        self.last_actions.clear()
