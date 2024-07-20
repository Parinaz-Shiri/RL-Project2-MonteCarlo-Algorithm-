import tkinter as tk
import numpy as np
import random

class GridWorld:
    def __init__(self, master):
        self.master = master
        self.master.title("GridWorld")

        self.grid_size = 5
        self.cell_size = 100

        canvas_height = self.grid_size * self.cell_size + 50
        self.canvas = tk.Canvas(master, width=self.grid_size * self.cell_size, height=canvas_height)
        self.canvas.pack()

        self.rewards = np.zeros((self.grid_size, self.grid_size))
        self.transitions = {}

        self.rewards[0, 1] = 5  # Blue
        self.rewards[0, 4] = 2.5  # Green
        self.transitions[(0, 1)] = (3, 2)  # Move to RED

        self.actions = ['up', 'down', 'left', 'right']
        self.action_probs = {'up': 0.25, 'down': 0.25, 'left': 0.25, 'right': 0.25}

        self.values = np.zeros((self.grid_size, self.grid_size))  # Value function, initializing all values of states to be 0
        self.policy = np.full((self.grid_size, self.grid_size), ' ')  # Policy array

        self.highest_value_label = tk.Label(self.master, text="")
        self.highest_value_label.pack()

        # Dropdown menu to select evaluation method
        self.method_var = tk.StringVar(master)
        self.method_var.set("Monte Carlo with stochastic environment")
        self.method_menu = tk.OptionMenu(master, self.method_var, "Fixed Start Monte Carlo Stochastic Environment")
        self.method_menu.pack()

        self.start_button = tk.Button(master, text="Start", command=self.start_evaluation)
        self.start_button.pack()

        self.reset_button = tk.Button(master, text="Reset", command=self.reset_values)
        self.reset_button.pack()

        self.initialize_values()
        self.draw_grid()
        self.update_policy_display()

    def initialize_values(self):
        self.green_pos = (0, 4)
        self.blue_pos = (0, 1)
        self.terminal_states = [(2, 4), (4, 0)]

    def draw_grid(self):  # colors for special states
        self.colors = {
            self.blue_pos: "blue",
            self.green_pos: "green",
            (4, 2): "red",
            (4, 4): "yellow"
        }

        for i in range(self.grid_size):
            for j in range(self.grid_size):
                x0 = j * self.cell_size
                y0 = i * self.cell_size
                x1 = x0 + self.cell_size
                y1 = y0 + self.cell_size
                if (i, j) in self.terminal_states:
                    color = "black"
                else:
                    color = self.colors.get((i, j), "white")
                self.canvas.create_rectangle(x0, y0, x1, y1, fill=color, outline="black")

    def update_policy_display(self):
        self.canvas.delete("policy")

        arrow_offsets = {
            'U': (0, -0.5),
            'D': (0, 0.5),
            'L': (-0.5, 0),
            'R': (0.5, 0)
        }

        arrow_length = self.cell_size * 0.4

        for i in range(self.grid_size):
            for j in range(self.grid_size):
                x0 = j * self.cell_size + self.cell_size / 2
                y0 = i * self.cell_size + self.cell_size / 2
                action = self.policy[i, j]
                if action in arrow_offsets:  # Ensure the action is valid
                    dx, dy = arrow_offsets[action]
                    x1 = x0 + dx * arrow_length
                    y1 = y0 + dy * arrow_length
                    self.canvas.create_line(x0, y0, x1, y1, tags="policy", arrow="last", fill="black", width=2)

    def get_next_state(self, i, j, action):
        ni, nj = i, j
        if action == 'up':
            ni -= 1
        elif action == 'down':
            ni += 1
        elif action == 'left':
            nj -= 1
        elif action == 'right':
            nj += 1

        # Define transitions and rewards clearly
        if (i, j) == self.blue_pos:
            return 3, 2, 5.0  # Special state
        elif (i, j) == self.green_pos:
            # Move to (4, 2) or (4, 4) with equal probability
            if np.random.rand() < 0.5:
                return 4, 2, 2.5
            else:
                return 4, 4, 2.5
        elif ni < 0 or ni >= self.grid_size or nj < 0 or nj >= self.grid_size:
            return i, j, -0.5  # Stay in the same position, negative reward
        elif (ni, nj) == (2, 4) or (ni, nj) == (4, 0):
            return ni, nj, 0.0  # Terminal states
        else:
            return ni, nj, -0.2  # Normal state, -0.2 reward

    def permute_green_blue_positions(self):
        if np.random.rand() < 0.1:
            self.green_pos, self.blue_pos = self.blue_pos, self.green_pos
            self.draw_grid()

    def generate_episode(self):
        episode = []
        state = (0, 0)  # Start from a fixed initial state

        while state not in self.terminal_states:
            action = random.choices(self.actions, weights=[self.action_probs[a] for a in self.actions])[0]
            ni, nj, reward = self.get_next_state(*state, action)
            episode.append((state, action, reward))
            state = (ni, nj)
            self.permute_green_blue_positions()

        return episode

    def update_policy(self, returns, N, epsilon=0.1):
        for state in returns:
            max_return = max(returns[state].values())
            best_action = random.choice([action for action, value in returns[state].items() if value == max_return])
            for action in self.actions:
                if action == best_action:
                    self.action_probs[action] = 1 - epsilon + (epsilon / len(self.actions))
                else:
                    self.action_probs[action] = epsilon / len(self.actions)
            self.policy[state] = best_action[0].upper()  # Update the policy for this state

    def Monte_Carlo(self, gamma=0.95, epsilon=0.1):
        print("Running Monte Carlo")
        returns = {}
        N = {}

        for i in range(self.grid_size):
            for j in range(self.grid_size):
                returns[(i, j)] = {action: 0 for action in self.actions}
                N[(i, j)] = {action: 0 for action in self.actions}

        for _ in range(10000):  # Number of episodes
            episode = self.generate_episode()
            G = 0
            for t in reversed(range(len(episode))):
                state, action, reward = episode[t]
                G = gamma * G + reward
                if (state, action) not in [(x[0], x[1]) for x in episode[:t]]:
                    N[state][action] += 1
                    returns[state][action] += (G - returns[state][action]) / N[state][action]
                    self.values[state] += (G - self.values[state]) / N[state][action]

            self.update_policy(returns, N, epsilon)
            self.update_policy_display()  # Call to update policy display after each episode
            self.master.update()

        self.display_optimal_policy()

    def start_evaluation(self):
        self.reset_values()
        self.Monte_Carlo()

    def reset_values(self):
        self.initialize_values()
        self.canvas.delete("policy")
        self.highest_value_label.config(text="")
        self.policy = np.full((self.grid_size, self.grid_size), ' ')  # Reset policy

    def display_optimal_policy(self):
        print("Optimal Policy:")
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                print(f"State ({i}, {j}): {self.policy[i, j]}")
        self.update_policy_display()


if __name__ == "__main__":
    root = tk.Tk()
    grid_world = GridWorld(root)
    root.mainloop()
