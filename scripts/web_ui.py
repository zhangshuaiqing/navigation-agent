#!/usr/bin/env python3
"""
Gradio Web UI for Navigation Agent.

Run with:
    uv run python scripts/web_ui.py

Then open the displayed URL in your browser.
"""

import sys
import os
import time
import threading
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import ListedColormap

import gradio as gr

from src.env.gridworld import GridWorld, CellType
from src.agent.navigator import (
    create_heuristic_navigator,
    create_react_navigator,
    HeuristicNavigator,
    ReActNavigator,
)


# ═══════════════════════════════════════════════════════════════
# Global state
# ═══════════════════════════════════════════════════════════════

class AppState:
    def __init__(self):
        self.env: GridWorld = None
        self.agent = None
        self.agent_type: str = "heuristic"
        self.path_history: list = []
        self.step_count: int = 0
        self.total_reward: float = 0.0
        self.is_running: bool = False
        self.logs: list = []
        self.cell_colors = {
            CellType.EMPTY: "#f0f0f0",
            CellType.OBSTACLE: "#333333",
            CellType.AGENT: "#4CAF50",
            CellType.GOAL: "#FF5722",
            CellType.PATH: "#BBDEFB",
        }

    def reset(self, size, obstacle_ratio, seed, agent_type, llm_model, llm_provider, random_start_goal=False, observation_mode="full", view_range=1, num_dynamic=0, num_goals=1, task_type="sequential"):
        """Reset environment and agent."""
        seed_val = int(seed) if seed else None
        self.env = GridWorld(
            size=int(size),
            obstacle_ratio=float(obstacle_ratio),
            seed=seed_val,
            random_start_goal=random_start_goal,
            observation_mode=observation_mode,
            view_range=int(view_range),
            num_dynamic_obstacles=int(num_dynamic),
            num_goals=int(num_goals),
            task_type=task_type,
        )
        self.agent_type = agent_type
        self.path_history = [self.env.agent_pos]
        self.step_count = 0
        self.total_reward = 0.0
        self.is_running = False
        self.logs = ["Environment reset."]

        if agent_type == "llm":
            try:
                self.agent = create_react_navigator(
                    self.env,
                    llm_provider=llm_provider,
                    llm_model=llm_model,
                )
                self.logs.append(f"LLM: {llm_provider}/{llm_model}")
            except ValueError as e:
                self.logs.append(f"WARNING: {e}. Falling back to heuristic.")
                self.agent = create_heuristic_navigator(self.env, use_bfs_hint=True)
                self.agent_type = "heuristic"
        else:
            self.agent = create_heuristic_navigator(self.env, use_bfs_hint=True)

        self.logs.append(f"Agent: {type(self.agent).__name__}")
        self.logs.append(f"Start: {self.env.agent_pos}, Goal: {self.env.goal_pos}")

        return self.render_grid(), self.get_status(), self.get_logs()

    def step(self):
        """Execute one step."""
        if self.env is None or self.env.done:
            return self.render_grid(), self.get_status(), self.get_logs()

        obs = self.env._get_obs()
        action = self.agent.act(obs)
        obs, reward, done, info = self.env.step(action)

        self.step_count += 1
        self.total_reward += reward
        self.path_history.append(self.env.agent_pos)
        self.agent.post_step()

        log_msg = f"Step {self.step_count}: {action} | Reward: {reward:+.2f} | {info.get('reason', '')}"
        self.logs.append(log_msg)

        if done:
            if self.env.agent_pos == self.env.goal_pos:
                self.logs.append(f"SUCCESS! Reached goal in {self.step_count} steps!")
            else:
                self.logs.append(f"Failed: {info.get('reason', 'unknown')}")
            self.is_running = False

        return self.render_grid(), self.get_status(), self.get_logs()

    def run_episode(self, delay=0.3):
        """Run full episode with auto-play."""
        if self.env is None:
            return self.render_grid(), self.get_status(), self.get_logs()

        self.is_running = True
        max_steps = self.env.max_steps

        while self.is_running and not self.env.done and self.step_count < max_steps:
            obs = self.env._get_obs()
            action = self.agent.act(obs)
            obs, reward, done, info = self.env.step(action)

            self.step_count += 1
            self.total_reward += reward
            self.path_history.append(self.env.agent_pos)
            self.agent.post_step()

            log_msg = f"Step {self.step_count}: {action} | Reward: {reward:+.2f}"
            self.logs.append(log_msg)

            # Yield intermediate states for streaming updates
            yield self.render_grid(), self.get_status(), self.get_logs()

            if done:
                if self.env.agent_pos == self.env.goal_pos:
                    self.logs.append(f"SUCCESS! Reached goal in {self.step_count} steps!")
                else:
                    self.logs.append(f"Failed: {info.get('reason', 'unknown')}")
                self.is_running = False
                break

            time.sleep(delay)

        # Final yield
        yield self.render_grid(), self.get_status(), self.get_logs()

    def stop(self):
        """Stop auto-run."""
        self.is_running = False
        self.logs.append("Stopped by user.")
        return self.render_grid(), self.get_status(), self.get_logs()

    def render_grid(self):
        """Render the grid as a matplotlib figure."""
        if self.env is None:
            return None

        fig, ax = plt.subplots(figsize=(8, 8))
        size = self.env.size

        # Draw cells
        for r in range(size):
            for c in range(size):
                # Fog of war: unvisited cells are hidden
                is_fog = (
                    self.env.observation_mode == "fog_of_war"
                    and not self.env.visited_mask[r, c]
                )
                
                if is_fog:
                    color = "#CCCCCC"  # Gray fog
                    label = ""
                elif (r, c) == self.env.agent_pos:
                    color = self.cell_colors[CellType.AGENT]
                    label = "A"
                elif (r, c) == self.env.goal_pos:
                    color = self.cell_colors[CellType.GOAL]
                    label = "G"
                elif self.env.grid[r, c] == CellType.DYNAMIC_OBSTACLE:
                    color = "#FF9800"  # Orange for dynamic obstacles
                    label = "D"
                elif self.env.task and (r, c) in self.env.all_goals:
                    idx = self.env.all_goals.index((r, c))
                    is_completed = (
                        idx < self.env.task.active_idx if self.env.task.type == "sequential"
                        else idx in self.env.task.completed_goals
                    )
                    if (r, c) == self.env.goal_pos and not is_completed:
                        color = "#FF5722"  # Active goal - bright orange
                        label = "G"
                    elif is_completed:
                        color = "#4CAF50"  # Completed - green
                        label = "C"
                    else:
                        color = "#BF360C"  # Pending - dark orange
                        label = "g"
                elif (r, c) in self.path_history[:-1]:
                    color = self.cell_colors[CellType.PATH]
                    label = ""
                elif self.env.grid[r, c] == CellType.OBSTACLE:
                    color = self.cell_colors[CellType.OBSTACLE]
                    label = ""
                else:
                    color = self.cell_colors[CellType.EMPTY]
                    label = ""

                rect = patches.Rectangle(
                    (c, size - 1 - r), 1, 1,
                    linewidth=1, edgecolor="#888888", facecolor=color
                )
                ax.add_patch(rect)

                if label:
                    ax.text(
                        c + 0.5, size - 1 - r + 0.5, label,
                        ha="center", va="center",
                        fontsize=20, fontweight="bold", color="white"
                    )

        # Draw path arrows
        if len(self.path_history) > 1:
            for i in range(len(self.path_history) - 1):
                r1, c1 = self.path_history[i]
                r2, c2 = self.path_history[i + 1]
                ax.annotate(
                    "",
                    xy=(c2 + 0.5, size - 1 - r2 + 0.5),
                    xytext=(c1 + 0.5, size - 1 - r1 + 0.5),
                    arrowprops=dict(arrowstyle="->", color="#2196F3", lw=2),
                )

        ax.set_xlim(0, size)
        ax.set_ylim(0, size)
        ax.set_aspect("equal")
        ax.axis("off")

        plt.tight_layout()
        return fig

    def get_status(self):
        """Get status string."""
        if self.env is None:
            return "No environment. Click Reset to start."

        status = f"""Agent: {type(self.agent).__name__}
Position: {self.env.agent_pos}
Goal: {self.env.goal_pos}
Distance: {abs(self.env.agent_pos[0] - self.env.goal_pos[0]) + abs(self.env.agent_pos[1] - self.env.goal_pos[1])}
Steps: {self.step_count}
Total Reward: {self.total_reward:.2f}
Status: {'DONE - Goal Reached!' if self.env.agent_pos == self.env.goal_pos else 'DONE - Max Steps' if self.env.done else 'Running'}
"""
        return status

    def get_logs(self):
        """Get logs as string."""
        return "\n".join(self.logs[-50:])  # Keep last 50 logs


# Global state instance
state = AppState()


# ═══════════════════════════════════════════════════════════════
# Gradio UI
# ═══════════════════════════════════════════════════════════════

def create_ui():
    with gr.Blocks(title="Navigation Agent - GridWorld", theme=gr.themes.Soft()) as demo:
        gr.Markdown("""
        # Navigation Agent - GridWorld Demo
        Navigate an agent through a grid world using LangChain-powered agents.
        """)

        with gr.Row():
            # ── Left: Controls ──────────────────────────────────
            with gr.Column(scale=1):
                gr.Markdown("### Controls")

                grid_size = gr.Slider(
                    minimum=4, maximum=20, value=8, step=1,
                    label="Grid Size"
                )
                obstacle_ratio = gr.Slider(
                    minimum=0.0, maximum=0.5, value=0.2, step=0.05,
                    label="Obstacle Ratio"
                )
                seed_input = gr.Number(
                    value=42, label="Random Seed (0 = random)", precision=0
                )
                agent_type = gr.Radio(
                    choices=["heuristic", "llm"],
                    value="heuristic",
                    label="Agent Type"
                )
                llm_model = gr.Dropdown(
                    choices=["gpt-4o-mini", "gpt-4o", "gpt-3.5-turbo",
                             "deepseek-chat", "deepseek-reasoner",
                             "moonshot-v1-8k", "moonshot-v1-32k"],
                    value="gpt-4o-mini",
                    label="LLM Model"
                )
                llm_provider = gr.Dropdown(
                    choices=["openai", "deepseek", "kimi"],
                    value="openai",
                    label="LLM Provider"
                )
                random_sg = gr.Checkbox(
                    value=False,
                    label="Random Start / Goal"
                )
                obs_mode = gr.Dropdown(
                    choices=["full", "local", "fog_of_war"],
                    value="full",
                    label="Observation Mode"
                )
                view_range = gr.Slider(
                    minimum=1, maximum=5, value=1, step=1,
                    label="View Range"
                )
                num_dynamic = gr.Slider(
                    minimum=0, maximum=10, value=0, step=1,
                    label="Dynamic Obstacles"
                )
                num_goals = gr.Slider(
                    minimum=1, maximum=10, value=1, step=1,
                    label="Number of Goals"
                )
                task_type = gr.Dropdown(
                    choices=["sequential", "any_order", "collect"],
                    value="sequential",
                    label="Task Type"
                )
                delay_slider = gr.Slider(
                    minimum=0.05, maximum=2.0, value=0.3, step=0.05,
                    label="Auto-run Delay (seconds)"
                )

                with gr.Row():
                    reset_btn = gr.Button("Reset", variant="primary", size="lg")
                    step_btn = gr.Button("Step", variant="secondary", size="lg")

                with gr.Row():
                    run_btn = gr.Button("Run", variant="primary", size="lg")
                    stop_btn = gr.Button("Stop", variant="stop", size="lg")

                gr.Markdown("""
                ---
                **Legend:**
                - A (green) = Agent
                - G (bright orange) = Active Goal
                - g (dark orange) = Pending Goal
                - C (green) = Completed Goal
                - D (amber) = Dynamic Obstacle
                - Blue arrows = Path taken
                - Dark gray = Static Obstacle
                - Light blue = Visited cells
                """)

            # ── Right: Visualization ────────────────────────────
            with gr.Column(scale=2):
                grid_plot = gr.Plot(
                    label="Grid World",
                    value=state.render_grid() if state.env else None
                )
                status_text = gr.Textbox(
                    label="Status",
                    lines=7,
                    value="Click Reset to start."
                )
                log_output = gr.Textbox(
                    label="Logs",
                    lines=10,
                    value="",
                    autoscroll=True
                )

        # ── Event handlers ────────────────────────────────────
        def on_reset(size, obs_ratio, seed, a_type, llm, llm_provider, random_sg, obs_mode, view_range, num_dynamic, num_goals, task_type):
            s = int(seed) if seed and seed > 0 else None
            return state.reset(size, obs_ratio, s, a_type, llm, llm_provider, random_sg, obs_mode, view_range, num_dynamic, num_goals, task_type)

        def on_step():
            return state.step()

        def on_run(delay):
            for result in state.run_episode(delay):
                yield result

        def on_stop():
            return state.stop()

        reset_btn.click(
            fn=on_reset,
            inputs=[grid_size, obstacle_ratio, seed_input, agent_type, llm_model, llm_provider, random_sg, obs_mode, view_range, num_dynamic, num_goals, task_type],
            outputs=[grid_plot, status_text, log_output]
        )

        step_btn.click(
            fn=on_step,
            inputs=[],
            outputs=[grid_plot, status_text, log_output]
        )

        run_btn.click(
            fn=on_run,
            inputs=[delay_slider],
            outputs=[grid_plot, status_text, log_output]
        )

        stop_btn.click(
            fn=on_stop,
            inputs=[],
            outputs=[grid_plot, status_text, log_output]
        )

    return demo


if __name__ == "__main__":
    demo = create_ui()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True,
    )
