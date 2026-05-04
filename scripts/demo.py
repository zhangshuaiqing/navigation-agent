#!/usr/bin/env python3
"""
Navigation Agent Demo Script

Run with:
    uv run python scripts/demo.py

For LLM-powered agent:
    uv run python scripts/demo.py --use-llm --llm-provider openai
    uv run python scripts/demo.py --use-llm --llm-provider deepseek
    uv run python scripts/demo.py --use-llm --llm-provider kimi

Benchmark mode:
    uv run python scripts/demo.py --benchmark 100
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.env.gridworld import GridWorld
from src.agent.navigator import create_heuristic_navigator, create_react_navigator, LLM_PROVIDERS
from src.utils.viz import render_episode, benchmark_agent


def parse_args():
    parser = argparse.ArgumentParser(description="Navigation Agent Demo")
    parser.add_argument("--size", type=int, default=8, help="Grid size (default: 8)")
    parser.add_argument("--obstacles", type=float, default=0.2, help="Obstacle ratio (default: 0.2)")
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    parser.add_argument("--delay", type=float, default=0.3, help="Step delay in seconds (default: 0.3)")
    parser.add_argument("--use-llm", action="store_true", help="Use LLM-powered ReAct agent")
    parser.add_argument("--llm-provider", type=str, default="openai", choices=["openai", "deepseek", "kimi", "ollama", "vllm", "custom"], help="LLM provider")
    parser.add_argument("--llm-model", type=str, default=None, help="LLM model name (defaults to provider default)")
    parser.add_argument("--benchmark", type=int, default=None, help="Run benchmark over N episodes")
    parser.add_argument("--obs-mode", type=str, default="full", choices=["full", "local", "fog_of_war"], help="Observation mode")
    parser.add_argument("--view-range", type=int, default=1, help="View range for local/fog modes")
    parser.add_argument("--dynamic", type=int, default=0, help="Number of dynamic obstacles")
    parser.add_argument("--goals", type=int, default=1, help="Number of goals (multi-goal task)")
    parser.add_argument("--task-type", type=str, default="sequential", choices=["sequential", "any_order", "collect"], help="Task type for multi-goal")
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
    print(f"Goals: {args.goals}, Task Type: {args.task_type}")
    print(f"Agent: {'LLM ReAct (' + (args.llm_model or LLM_PROVIDERS[args.llm_provider]['default_model']) + ' via ' + args.llm_provider + ')' if args.use_llm else 'Heuristic/BFS'}")
    print("=" * 60)
    
    env = GridWorld(
        size=args.size,
        obstacle_ratio=args.obstacles,
        seed=args.seed,
        random_start_goal=args.random_start_goal,
        observation_mode=args.obs_mode,
        view_range=args.view_range,
        num_dynamic_obstacles=args.dynamic,
        num_goals=args.goals,
        task_type=args.task_type,
    )
    
    if args.use_llm:
        from src.agent.navigator import LLM_PROVIDERS
        provider_cfg = LLM_PROVIDERS.get(args.llm_provider, {})
        model_name = args.llm_model or provider_cfg.get("default_model", "unknown")
        print(f"\nINFO: Using LLM provider '{args.llm_provider}' with model '{model_name}'")
        print(f"INFO: Set {provider_cfg.get('env_key', 'OPENAI_API_KEY')} environment variable\n")
        
        agent = create_react_navigator(env, llm_provider=args.llm_provider, llm_model=args.llm_model)
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
