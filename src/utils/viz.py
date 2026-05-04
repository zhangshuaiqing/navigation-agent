"""
Visualization utilities for GridWorld navigation.
"""

import time
from typing import List, Optional, Tuple
from ..env.gridworld import GridWorld
from ..agent.navigator import NavigationAgent


def clear_screen():
    """Clear terminal screen."""
    print("\033[2J\033[H", end="")


def render_episode(
    env: GridWorld,
    agent: NavigationAgent,
    max_steps: Optional[int] = None,
    delay: float = 0.3,
    verbose: bool = True,
    use_llm: bool = False,
) -> dict:
    """
    Run a full episode and render it.
    
    Returns stats dict with steps, reward, success.
    """
    if max_steps is None:
        max_steps = env.max_steps
    
    total_reward = 0.0
    path_taken = [env.agent_pos]
    step = 0
    
    if verbose:
        clear_screen()
        print("=" * 50)
        print("     GridWorld Navigation Agent Demo")
        print("=" * 50)
        print(f"Grid size: {env.size}x{env.size}")
        print(f"Start: {env.agent_pos}, Goal: {env.goal_pos}")
        print(f"Agent type: {'LLM ReAct' if use_llm else 'Heuristic/BFS'}")
        print("-" * 50)
        print(env.render())
        print("-" * 50)
        time.sleep(delay)
    
    while step < max_steps and not env.done:
        obs = env._get_obs()
        action = agent.act(obs)
        
        # Show LLM thought if available (ReAct agent)
        thought = None
        if hasattr(agent, 'get_last_thought'):
            thought = agent.get_last_thought()
        
        obs, reward, done, info = env.step(action)
        total_reward += reward
        path_taken.append(env.agent_pos)
        agent.post_step()
        step += 1
        
        if verbose:
            clear_screen()
            print("=" * 50)
            print("     GridWorld Navigation Agent Demo")
            print("=" * 50)
            print(f"Step {step}/{max_steps} | Action: {action:6s} | Reward: {reward:+.2f}")
            print(f"Total reward: {total_reward:.2f} | Distance: {obs['distance_to_goal']}")
            print(f"Reason: {info.get('reason', 'N/A')}")
            if thought:
                thought_str = thought.get("thought", "")
                tools = thought.get("tools", [])
                if thought_str:
                    # Truncate long thoughts
                    if len(thought_str) > 200:
                        thought_str = thought_str[:200] + "..."
                    print(f"LLM: {thought_str}")
                if tools:
                    print(f"Tools: {', '.join(tools)}")
            print("-" * 50)
            print(env.render(mark_path=path_taken[:-1]))
            print("-" * 50)
            time.sleep(delay)
        
        if done:
            break
    
    success = env.agent_pos == env.goal_pos
    
    if verbose:
        print("\n" + "=" * 50)
        if success:
            print(f"\n\U0001f389 SUCCESS! Reached goal in {step} steps!")
        else:
            print(f"\n\u274c Failed to reach goal. Steps: {step}")
        print(f"Total reward: {total_reward:.2f}")
        print(f"Path length: {len(path_taken)}")
        print("=" * 50)
    
    return {
        "success": success,
        "steps": step,
        "total_reward": total_reward,
        "path": path_taken,
        "final_pos": env.agent_pos,
    }


def run_interactive_demo(
    size: int = 8,
    obstacle_ratio: float = 0.2,
    seed: Optional[int] = None,
    delay: float = 0.3,
    use_llm: bool = False,
    llm_model: str = "gpt-4o-mini",
) -> dict:
    """
    Run an interactive demo with the specified configuration.
    
    Args:
        size: Grid size
        obstacle_ratio: Ratio of obstacles in the grid
        seed: Random seed
        delay: Delay between steps in seconds
        use_llm: Whether to use LLM-based agent (requires OPENAI_API_KEY)
        llm_model: LLM model name if use_llm=True
    
    Returns:
        Episode statistics
    """
    from ..agent.navigator import create_heuristic_navigator, create_react_navigator
    
    env = GridWorld(size=size, obstacle_ratio=obstacle_ratio, seed=seed)
    
    if use_llm:
        import os
        if not os.environ.get("OPENAI_API_KEY"):
            print("WARNING: OPENAI_API_KEY not set. Falling back to heuristic agent.")
            agent = create_heuristic_navigator(env, use_bfs_hint=True)
            use_llm = False
        else:
            from langchain_openai import ChatOpenAI
            llm = ChatOpenAI(model=llm_model, temperature=0)
            agent = create_react_navigator(env, llm=llm)
    else:
        agent = create_heuristic_navigator(env, use_bfs_hint=True)
    
    return render_episode(
        env=env,
        agent=agent,
        delay=delay,
        verbose=True,
        use_llm=use_llm,
    )


def benchmark_agent(
    env: GridWorld,
    agent: NavigationAgent,
    num_episodes: int = 10,
) -> dict:
    """Benchmark an agent over multiple episodes."""
    successes = 0
    total_steps = 0
    total_reward = 0.0
    
    for ep in range(num_episodes):
        env.reset(new_map=True)
        agent.reset()
        
        stats = render_episode(
            env=env,
            agent=agent,
            verbose=False,
            use_llm=isinstance(agent, type(agent)) and "ReAct" in type(agent).__name__,
        )
        
        if stats["success"]:
            successes += 1
        total_steps += stats["steps"]
        total_reward += stats["total_reward"]
        
        emoji = "\u2705" if stats['success'] else "\u274c"
        print(f"Episode {ep+1}/{num_episodes}: "
              f"{emoji} "
              f"Steps: {stats['steps']:3d}, Reward: {stats['total_reward']:+.2f}")
    
    print(f"\nBenchmark Results ({num_episodes} episodes):")
    print(f"  Success rate: {successes}/{num_episodes} ({successes/num_episodes*100:.1f}%)")
    print(f"  Avg steps: {total_steps/num_episodes:.1f}")
    print(f"  Avg reward: {total_reward/num_episodes:.2f}")
    
    return {
        "success_rate": successes / num_episodes,
        "avg_steps": total_steps / num_episodes,
        "avg_reward": total_reward / num_episodes,
    }
