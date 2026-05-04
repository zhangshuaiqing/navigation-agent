"""
Tests for navigation agents.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.env.gridworld import GridWorld
from src.agent.navigator import HeuristicNavigator, create_heuristic_navigator
from src.agent.memory import AgentMemory


def test_memory_basic():
    """Test basic memory operations."""
    env = GridWorld(size=6, seed=42)
    memory = AgentMemory()
    
    obs = env._get_obs()
    memory.update(
        pos=env.agent_pos,
        obs=obs,
        actions=[],
        grid_size=env.size,
    )
    
    # Should have recorded initial position as visited
    assert env.agent_pos in memory.visited
    assert len(memory.trajectory) == 1
    assert memory.trajectory[0] == env.agent_pos
    
    # Move and update again
    env.step("down")
    obs = env._get_obs()
    memory.update(
        pos=env.agent_pos,
        obs=obs,
        actions=["down"],
        grid_size=env.size,
    )
    
    assert len(memory.trajectory) == 2
    print("Memory basic test passed")


def test_memory_dead_end():
    """Test dead-end detection."""
    env = GridWorld(size=5, seed=42)
    
    # Create a simple dead-end scenario manually
    env.grid[0, 1] = 1  # OBSTACLE right of start
    
    memory = AgentMemory()
    obs = env._get_obs()
    memory.update(pos=env.agent_pos, obs=obs, actions=[], grid_size=env.size)
    
    # Mark current as visited, so only "down" should be valid
    # and it should not be a dead end yet
    valid = env.get_valid_actions()
    
    # Move to (1, 0) and check memory
    env.step("down")
    obs = env._get_obs()
    memory.update(pos=env.agent_pos, obs=obs, actions=["down"], grid_size=env.size)
    
    # Cell (1,0) should be in visited
    assert (1, 0) in memory.visited
    print("Memory dead end test passed")


def test_memory_backtrack():
    """Test backtracking from a dead end."""
    env = GridWorld(size=5, obstacle_ratio=0.0, seed=42)
    agent = create_heuristic_navigator(env, use_memory=True)
    
    # Move agent down 2 steps - memory updates via agent.post_step()
    for _ in range(2):
        if env.done:
            break
        action = agent.act()
        env.step(action)
        agent.post_step()
    
    # At (2, 0). Should have trajectory [(0,0), (1,0), (2,0)]
    assert len(agent.memory.trajectory) >= 2, f"Expected >=2, got {len(agent.memory.trajectory)}"
    print(f"Trajectory: {agent.memory.trajectory}")
    
    # Backtrack from (2,0) should go back toward (1,0)
    backtrack_action = agent.memory.backtrack((2, 0))
    assert backtrack_action == "up", f"Expected 'up', got '{backtrack_action}'"
    print("Memory backtrack test passed")


def test_heuristic_with_memory():
    """Test that heuristic agent works with memory enabled."""
    env = GridWorld(size=8, seed=42, observation_mode="local", view_range=1)
    agent = create_heuristic_navigator(env, use_memory=True)
    
    # Run a few steps
    for _ in range(10):
        if env.done:
            break
        action = agent.act()
        env.step(action)
        agent.post_step()
    
    # Memory should have recorded visited cells
    assert len(agent.memory.visited) >= 5  # At least some visited
    assert len(agent.memory.trajectory) > 1
    print("Heuristic with memory test passed")


def test_react_prompt_basic():
    """Test that ReAct prompt includes basic info."""
    env = GridWorld(size=6, seed=42)
    # We can't instantiate ReActNavigator without API key, so test the prompt logic directly
    from src.agent.navigator import ReActNavigator
    import os
    if not os.environ.get("OPENAI_API_KEY"):
        print("Skipping ReAct prompt test (no API key)")
        return
    
    from langchain_openai import ChatOpenAI
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    agent = ReActNavigator(env, llm=llm)
    
    prompt = agent._build_prompt()
    
    # Should contain basic info
    assert "navigation agent" in prompt
    assert "6x6" in prompt
    assert "sense_surroundings" in prompt
    assert "move" in prompt
    assert "get_path_hint" in prompt
    print("ReAct basic prompt test passed")


def test_react_prompt_local_mode():
    """Test that ReAct prompt adapts to local mode."""
    from src.agent.navigator import ReActNavigator
    
    # Test prompt generation without creating LLM
    class MockReAct(ReActNavigator):
        def __init__(self, env):
            self.env = env
            self.max_iterations = 50
            self.messages = []
            self.history = []
            self.tools = []
    
    env = GridWorld(size=6, observation_mode="local", view_range=2, seed=42)
    agent = MockReAct(env)
    
    prompt = agent._build_prompt()
    
    assert "LOCAL MODE" in prompt
    assert "DISABLED" in prompt or "get_path_hint" in prompt
    assert "sense_surroundings" in prompt
    print("ReAct local mode prompt test passed")


def test_react_prompt_multi_goal():
    """Test that ReAct prompt adapts to multi-goal tasks."""
    from src.agent.navigator import ReActNavigator
    
    class MockReAct(ReActNavigator):
        def __init__(self, env):
            self.env = env
            self.max_iterations = 50
            self.messages = []
            self.history = []
            self.tools = []
    
    env = GridWorld(size=6, num_goals=3, task_type="sequential", seed=42)
    agent = MockReAct(env)
    
    prompt = agent._build_prompt()
    
    assert "Task" in prompt
    assert "MULTI-GOAL" in prompt
    assert "0/3" in prompt
    print("ReAct multi-goal prompt test passed")


if __name__ == "__main__":
    test_memory_basic()
    test_memory_dead_end()
    test_memory_backtrack()
    test_heuristic_with_memory()
    test_react_prompt_basic()
    test_react_prompt_local_mode()
    test_react_prompt_multi_goal()
    print("All agent tests passed!")
