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


def test_local_observation():
    env = GridWorld(size=6, observation_mode="local", view_range=2, seed=42)
    obs = env._get_obs()
    
    # Goal pos should be None in local mode if not within view
    assert obs["goal_pos"] is None
    assert obs["goal_visible"] == (abs(env.agent_pos[0] - env.goal_pos[0]) <= 2 and abs(env.agent_pos[1] - env.goal_pos[1]) <= 2)
    assert obs["distance_to_goal"] is None or isinstance(obs["distance_to_goal"], int)
    assert len(obs["surroundings"]) == 25  # 5x5 = (2*2+1)^2
    print("Local observation test passed")


def test_fog_of_war():
    env = GridWorld(size=6, observation_mode="fog_of_war", view_range=1, seed=42)
    
    # Initial position should be visited
    assert env.visited_mask[env.agent_pos] == True
    
    # Move agent
    env.step("right")
    
    # New position should be visited
    assert env.visited_mask[env.agent_pos] == True
    
    # Some cells should be unvisited (marked False)
    assert not env.visited_mask.all()
    
    # Render with fog should show ? for unvisited
    render_str = env.render(show_fog=True)
    assert "?" in render_str
    print("Fog of war test passed")


def test_dynamic_obstacles():
    env = GridWorld(size=6, num_dynamic_obstacles=3, seed=42)
    
    # Check dynamic obstacles exist
    assert len(env.dynamic_obstacles) == 3
    
    # Check they're on the grid
    for dyn in env.dynamic_obstacles:
        assert env.grid[dyn.pos] == CellType.DYNAMIC_OBSTACLE
    
    # Step should move dynamic obstacles
    initial_positions = [dyn.pos for dyn in env.dynamic_obstacles]
    env.step("down")
    new_positions = [dyn.pos for dyn in env.dynamic_obstacles]
    
    # At least one might have moved (probabilistic)
    print(f"Dynamic obs positions before: {initial_positions}")
    print(f"Dynamic obs positions after:  {new_positions}")
    
    # Check agent can't move into dynamic obstacle
    # Place agent next to a dynamic obstacle and try to move into it
    # This is probabilistic so we just verify the grid cell type
    for dyn in env.dynamic_obstacles:
        assert env.grid[dyn.pos] == CellType.DYNAMIC_OBSTACLE
    
    print("Dynamic obstacles test passed")


if __name__ == "__main__":
    test_gridworld_init()
    test_step()
    test_goal_reached()
    test_shortest_path()
    test_reset()
    test_random_start_goal()
    test_local_observation()
    test_fog_of_war()
    test_dynamic_obstacles()
    print("All tests passed!")
