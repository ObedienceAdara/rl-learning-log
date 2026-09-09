from __future__ import annotations

from pathlib import Path
import tkinter as tk
from tkinter import messagebox

import numpy as np
from PIL import Image, ImageTk

from src.gridworld import ARROWS, COLS, GOAL, OBSTACLES, ROWS, START, state_to_position, step

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "q_table_500_episodes.npy"
SPRITE_PATH = ROOT / "assets" / "robot_sprite.png"
SPRITESHEET_PATH = ROOT / "assets" / "robot_spritesheet.png"

CELL_SIZE = 150
GRID_X = 30
GRID_Y = 130
WINDOW_W = GRID_X * 2 + COLS * CELL_SIZE
WINDOW_H = 700
ANIMATION_MS = 550


class GridWorldGUI:
    def __init__(self, root: tk.Tk, q_table: np.ndarray):
        self.root = root
        self.q_table = q_table
        self.state = START
        self.path = self.greedy_path()
        self.path_index = 0
        self.running = False

        root.title("Q-Learning Grid World — 500 Episodes")
        root.geometry(f"{WINDOW_W}x{WINDOW_H}")
        root.resizable(False, False)

        self.canvas = tk.Canvas(
            root,
            width=WINDOW_W,
            height=WINDOW_H,
            bg="#101418",
            highlightthickness=0,
        )
        self.canvas.pack(fill="both", expand=True)

        self.sprite = self.load_sprite()
        self.draw_static()
        self.draw_agent(self.state)

        controls = tk.Frame(root, bg="#101418")
        controls.place(x=GRID_X, y=660, width=COLS * CELL_SIZE, height=35)
        tk.Button(controls, text="Run optimal path", command=self.start_animation).pack(side="left")
        tk.Button(controls, text="Reset", command=self.reset).pack(side="right")

        root.after(500, self.start_animation)

    def load_sprite(self):
        candidate = SPRITE_PATH if SPRITE_PATH.exists() else SPRITESHEET_PATH
        if not candidate.exists():
            return None

        image = Image.open(candidate).convert("RGBA")

        if candidate.name == "robot_spritesheet.png":
            cols = 5
            rows = 5
            frame_w = image.width // cols
            frame_h = image.height // rows
            image = image.crop((0, 0, frame_w, frame_h))

        image.thumbnail((CELL_SIZE - 26, CELL_SIZE - 26), Image.Resampling.NEAREST)
        return ImageTk.PhotoImage(image)

    def draw_static(self):
        self.canvas.delete("static")
        self.canvas.create_text(
            WINDOW_W // 2,
            35,
            text="Tabular Q-Learning — Greedy Policy After 500 Episodes",
            fill="white",
            font=("Arial", 18, "bold"),
            tags="static",
        )
        self.canvas.create_text(
            WINDOW_W // 2,
            70,
            text="Start → navigate around obstacles → Goal",
            fill="#b8c1cc",
            font=("Arial", 11),
            tags="static",
        )

        for row in range(ROWS):
            for col in range(COLS):
                x1 = GRID_X + col * CELL_SIZE
                y1 = GRID_Y + row * CELL_SIZE
                x2 = x1 + CELL_SIZE
                y2 = y1 + CELL_SIZE
                state = row * COLS + col

                fill = "#202830"
                if state in OBSTACLES:
                    fill = "#7d2430"
                elif state == GOAL:
                    fill = "#216e4e"
                elif state == START:
                    fill = "#1d4e89"

                self.canvas.create_rectangle(
                    x1, y1, x2, y2,
                    fill=fill,
                    outline="#59636e",
                    width=2,
                    tags="static",
                )

                if state in OBSTACLES:
                    self.canvas.create_text(
                        (x1 + x2) // 2, (y1 + y2) // 2,
                        text="X", fill="white", font=("Arial", 30, "bold"), tags="static"
                    )
                elif state == GOAL:
                    self.canvas.create_text(
                        (x1 + x2) // 2, y1 + 35,
                        text="GOAL", fill="white", font=("Arial", 12, "bold"), tags="static"
                    )
                    self.canvas.create_text(
                        (x1 + x2) // 2, (y1 + y2) // 2 + 10,
                        text="+10", fill="white", font=("Arial", 22, "bold"), tags="static"
                    )
                elif state != START:
                    action = int(np.argmax(self.q_table[state]))
                    self.canvas.create_text(
                        (x1 + x2) // 2,
                        (y1 + y2) // 2,
                        text=ARROWS[action],
                        fill="#d5dde5",
                        font=("Arial", 32, "bold"),
                        tags="static",
                    )

                self.canvas.create_text(
                    x1 + 12, y1 + 13,
                    text=f"s{state}", fill="#aeb7c2", anchor="nw", font=("Arial", 9), tags="static"
                )

    def greedy_path(self, max_steps: int = 50) -> list[int]:
        state = START
        path = [state]
        visited = {state}

        for _ in range(max_steps):
            if state == GOAL:
                return path

            action = int(np.argmax(self.q_table[state]))
            transition = step(state, action)

            if transition.next_state == state or transition.next_state in visited:
                candidates = []
                for candidate_action in range(4):
                    candidate = step(state, candidate_action)
                    if candidate.next_state != state and candidate.next_state not in visited:
                        candidates.append(
                            (self.q_table[state, candidate_action], candidate_action, candidate)
                        )
                if not candidates:
                    break
                _, _, transition = max(candidates, key=lambda item: item[0])

            state = transition.next_state
            path.append(state)
            visited.add(state)

            if state == GOAL:
                return path

        return path

    def draw_agent(self, state: int):
        self.canvas.delete("agent")
        row, col = state_to_position(state)
        x = GRID_X + col * CELL_SIZE + CELL_SIZE // 2
        y = GRID_Y + row * CELL_SIZE + CELL_SIZE // 2

        if self.sprite:
            self.canvas.create_image(x, y, image=self.sprite, tags="agent")
        else:
            radius = 27
            self.canvas.create_oval(
                x - radius, y - radius, x + radius, y + radius,
                fill="#e6b84b", outline="white", width=2, tags="agent"
            )
            self.canvas.create_text(
                x, y, text="AI", fill="#101418", font=("Arial", 12, "bold"), tags="agent"
            )

    def start_animation(self):
        if self.running:
            return

        self.path = self.greedy_path()
        if len(self.path) < 2 or self.path[-1] != GOAL:
            messagebox.showwarning(
                "No complete path",
                "The current greedy policy did not reach the goal. Run train_q_learning.py again and retry.",
            )
            return

        self.running = True
        self.path_index = 0
        self.state = self.path[0]
        self.draw_agent(self.state)
        self.animate_step()

    def animate_step(self):
        if not self.running:
            return

        if self.path_index >= len(self.path) - 1:
            self.running = False
            self.canvas.create_text(
                WINDOW_W // 2,
                625,
                text=f"Goal reached in {len(self.path) - 1} moves",
                fill="#7ee2a8",
                font=("Arial", 14, "bold"),
                tags="agent",
            )
            return

        self.path_index += 1
        self.state = self.path[self.path_index]
        self.draw_agent(self.state)
        self.root.after(ANIMATION_MS, self.animate_step)

    def reset(self):
        self.running = False
        self.state = START
        self.path_index = 0
        self.canvas.delete("agent")
        self.draw_agent(self.state)


def main():
    if not DATA_PATH.exists():
        raise SystemExit("Q-table not found. Run: python train_q_learning.py")

    q_table = np.load(DATA_PATH)
    root = tk.Tk()
    GridWorldGUI(root, q_table)
    root.mainloop()


if __name__ == "__main__":
    main()
