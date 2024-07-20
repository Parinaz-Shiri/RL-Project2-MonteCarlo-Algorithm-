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
        self.transitions[(0, 1)] = (3, 2)  # Move to RED

        self.rewards[0, 3] = 2.5  # B
        self.transitions[(0, 3)] = (2, 3)  # Move to B

        self.actions = ['U', 'D', 'L', 'R']  # Up, Down, Left, Right
        self.action_probs = {'U': 0.25, 'D': 0.25, 'L': 0.25, 'R': 0.25}

        self.values = np.zeros((self.grid_size, self.grid_size))  # Value function, initializing all values of states to be 0
        self.policy = np.full((self.grid_size, self.grid_size), ' ')  # Policy array

        self.highest_value_label = tk.Label(self.master, text="")
        self.highest_value_label.pack()

        self.policy_label = tk.Label(self.master, text="")
        self.policy_label.pack()

        # Dropdown menu to select evaluation method
        self.method_var = tk.StringVar(master)
        self.method_var.set("Monte Carlo with Exploring Starts")
        self.method_menu = tk.OptionMenu(master, self.method_var, "Monte Carlo with Exploring Starts", "Monte Carlo without Exploring Starts")
        self.method_menu.pack()

        self.start_button = tk.Button(master, text="Start", command=self.start_evaluation)
        self.start_button.pack()

        self.reset_button = tk.Button(master, text="Reset", command=self.reset_values)
        self.reset_button.pack()

        self.draw_grid()
        self.update_policy_display()

    def draw_grid(self):  # colors for special states
        self.colors = {
            (0, 1): "blue",
            (0, 4): "green",
            (4, 2): "red",
            (4, 4): "yellow",
            (2, 4): "black",
            (4, 0): "black",
        }

        for i in range(self.grid_size):
            for j in range(self.grid_size):
                x0 = j * self.cell_size
                y0 = i * self.cell_size
                x1 = x0 + self.cell_size
                y1 = y0 + self.cell_size
                color = self.colors.get((i, j), "white")
                self.canvas.create_rectangle(x0, y0, x1, y1, fill=color, outline="black")

    def update_policy_display(self):
        self.canvas.delete("policy")
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                x0 = j * self.cell_size
                y0 = i * self.cell_size
                action = self.policy[i, j]
                if action == 'U':
                    arrow = '↑'
                elif action == 'D':
                    arrow = '↓'
                elif action == 'L':
                    arrow = '←'
                elif action == 'R':
                    arrow = '→'
                else:
                    arrow = ' '
                self.canvas.create_text(x0 + self.cell_size / 2, y0 + self.cell_size / 2,
                                        text=arrow, tags="policy", fill="black", font=("Helvetica", 24))

    def get_next_state(self, i, j, action):
        ni, nj = i, j
        if action == 'U':
            ni -= 1
        elif action == 'D':
            ni += 1
        elif action == 'L':
            nj -= 1
        elif action == 'R':
            nj += 1

        if (i, j) == (0, 1):
            return 3, 2, 5.0  # Special state
        elif (i, j) == (0, 4):
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

    def generate_episode(self, exploring_starts=True):
        episode = []
        if exploring_starts:
            state = (random.randint(0, self.grid_size - 1), random.randint(0, self.grid_size - 1))
        else:
            state = (0, 0)  # Start from a fixed initial state

        while state not in [(2, 4), (4, 0)]:
            action = random.choices(self.actions, weights=[self.action_probs[a] for a in self.actions])[0]
            ni, nj, reward = self.get_next_state(*state, action)
            episode.append((state, action, reward))
            state = (ni, nj)

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
            self.policy[state] = best_action  # Update the policy for this state

    def Monte_Carlo(self, exploring_starts=True, gamma=0.95, epsilon=0.1):
        print("Running Monte Carlo" + (" with Exploring Starts" if exploring_starts else " without Exploring Starts"))
        returns = {}
        N = {}

        for i in range(self.grid_size):
            for j in range(self.grid_size):
                returns[(i, j)] = {action: 0 for action in self.actions}
                N[(i, j)] = {action: 0 for action in self.actions}

        for _ in range(10000):  # Number of episodes
            episode = self.generate_episode(exploring_starts)
            G = 0
            for t in reversed(range(len(episode))):
                state, action, reward = episode[t]
                G = gamma * G + reward
                if (state, action) not in [(x[0], x[1]) for x in episode[:t]]:
                    N[state][action] += 1
                    returns[state][action] += (G - returns[state][action]) / N[state][action]
                    self.values[state] += (G - self.values[state]) / N[state][action]

            self.update_policy(returns, N, epsilon)
            self.update_policy_display()
            self.master.update()

        self.display_optimal_policy()

    def start_evaluation(self):
        self.reset_values()
        method = self.method_var.get()
        if method == "Monte Carlo with Exploring Starts":
            self.Monte_Carlo(exploring_starts=True)
        elif method == "Monte Carlo without Exploring Starts":
            self.Monte_Carlo(exploring_starts=False)

    def reset_values(self):
        self.values = np.zeros((self.grid_size, self.grid_size))  # Reset values
        self.policy = np.full((self.grid_size, self.grid_size), ' ')  # Reset policy
        self.canvas.delete("policy")
        self.highest_value_label.config(text="")
        self.policy_label.config(text="")
        self.update_policy_display()

    def display_optimal_policy(self):
        method = self.method_var.get()
        self.policy_label.config(text=f"Optimal Policy of {method}")
        self.update_policy_display()


if __name__ == "__main__":
    root = tk.Tk()
    grid_world = GridWorld(root)
    root.mainloop()
