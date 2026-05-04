#!/usr/bin/env python3
"""
Navigation Agent Demo Script

Run with:
    uv run python scripts/demo.py

For LLM-powered agent (requires OPENAI_API_KEY):
    uv run python scripts/demo.py --use-llm

Benchmark mode:
    uv run python scripts/demo.py --benchmark 100
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.env.gridworld import GridWorld
from src.agent.navigator import create_heuristic_navigator, create_react_navigator
from src.utils.viz import render_episode, benchmark_agent


def parse_args():
    parser = argparse.ArgumentParser(description="Navigation Agent Demo")
    parser.add_argument("--size", type=int, default=8, help="Grid size (default: 8)")
    parser.add_argument("--obstacles", type=float, default=0.2, help="Obstacle ratio (default: 0.2)")
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    parser.add_argument("--delay", type=float, default=0.3, help="Step delay in seconds (default: 0.3)")
    parser.add_argument("--use-llm", action="store_true", help="Use LLM-powered ReAct agent")
    parser.add_argument("--llm-model", type=str, default="gpt-4o-mini", help="LLM model name")
    parser.add_argument("--benchmark", type=int, default=None, help="Run benchmark over N episodes")
    parser.add_argument("--obs-mode", type=str, default="full", choices=["full", "local", "fog_of_war"], help="Observation mode")
    parser.add_argument("--view-range", type=int, default=1, help="View range for local/fog modes")
    parser.add_argument("--dynamic", type=int, default=0, help="Number of dynamic obstacles")
    parser.add_argument("--max-steps", type=int, default=None, help="Max steps per episode")
    parser.add_argument("--random-start-goal", action="store_true", help="Randomize start and goal positions")
    parser.add_argument("--no-render", action="store_true", help="Disable rendering")
    return parser.parse_args()


def main():
    args = parse_args()
    
    print("=" * 60)
    print("   Navigation Agent - GridWorld Demo")
    print("=" * 60)
    print(f"Grid: {args.size}x{args.size}, Obstacles: {args.obstacles*100:.0f}%")
    print(f"Seed: {args.seed or 'random'}")
    print(f"Random Start/Goal: {args.random_start_goal}")
    print(f"Observation Mode: {args.obs_mode}, View Range: {args.view_range}")
    print(f"Dynamic Obstacles: {args.dynamic}")
    print(f"Agent: {'LLM ReAct (' + args.llm_model + ')' if args.use_llm else 'Heuristic/BFS'}")
    print("=" * 60)
    
    env = GridWorld(
        size=args.size,
        obstacle_ratio=args.obstacles,
        seed=args.seed,
        random_start_goal=args.random_start_goal,
        observation_mode=args.obs_mode,
        view_range=args.view_range,
        num_dynamic_obstacles=args.dynamic,
    )
    
    if args.use_llm:
        import os
        if not os.environ.get("OPENAI_API_KEY"):
            print("\nERROR: OPENAI_API_KEY not set!")
            print("Set it with: export OPENAI_API_KEY='your-key'")
            print("Or use heuristic agent without --use-llm\n")
            sys.exit(1)
        
        from langchain_openai import ChatOpenAI
        llm = ChatOpenAI(model=args.llm_model, temperature=0)
        agent = create_react_navigator(env, llm=llm)
    else:
        agent = create_heuristic_navigator(env, use_bfs_hint=True)
    
    if args.benchmark:
        print(f"\nRunning benchmark: {args.benchmark} episodes\n")
        benchmark_agent(env, agent, num_episodes=args.benchmark)
    else:
        stats = render_episode(
            env=env,
            agent=agent,
            max_steps=args.max_steps,
            delay=args.delay,
            verbose=not args.no_render,
            use_llm=args.use_llm,
        )
        
        if args.no_render:
            result_emoji = "\u2705" if stats['success'] else "\u274c"
            result_text = "SUCCESS" if stats['success'] else "FAILED"
            print(f"\nResult: {result_emoji} {result_text}")
            print(f"Steps: {stats['steps']}, Reward: {stats['total_reward']:+.2f}")


if __name__ == "__main__":
    main()
