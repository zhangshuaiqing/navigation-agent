"""
Tests for GridWorld environment.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.env.gridworld import GridWorld, CellType


def test_gridworld_init():
    env = GridWorld(size=5, obstacle_ratio=0.1, seed=42)
    assert env.size == 5
    assert env.agent_pos == (0, 0)
    assert env.goal_pos == (4, 4)
    assert env.grid.shape == (5, 5)


def test_step():
    env = GridWorld(size=5, obstacle_ratio=0.0, seed=42)
    obs, reward, done, info = env.step("down")
    assert env.agent_pos == (1, 0)
    assert not done


def test_goal_reached():
    env = GridWorld(size=3, obstacle_ratio=0.0, seed=42)
    # Move to goal at (2,2)
    env.step("down")
    env.step("down")
    env.step("right")
    env.step("right")
    assert env.agent_pos == (2, 2)
    assert env.done


def test_shortest_path():
    env = GridWorld(size=5, obstacle_ratio=0.0, seed=42)
    path = env.get_shortest_path()
    assert path is not None
    assert path[0] == (0, 0)
    assert path[-1] == (4, 4)


def test_reset():
    env = GridWorld(size=5, seed=42)
    env.step("down")
    env.reset()
    assert env.agent_pos == (0, 0)
    assert not env.done


def test_random_start_goal():
    env = GridWorld(size=6, random_start_goal=True, seed=42)
    assert env.agent_pos != (0, 0) or env.goal_pos != (5, 5)
    print(f"Random start: {env.agent_pos}, goal: {env.goal_pos}")
    
    # Run multiple times to verify randomness
    positions = set()
    for _ in range(10):
        env.reset(new_map=True)
        positions.add((env.agent_pos, env.goal_pos))
    assert len(positions) > 1, "Should produce different positions"
    print(f"Unique position pairs: {len(positions)}")


if __name__ == "__main__":
    test_gridworld_init()
    test_step()
    test_goal_reached()
    test_shortest_path()
    test_reset()
    test_random_start_goal()
    print("All tests passed!")
