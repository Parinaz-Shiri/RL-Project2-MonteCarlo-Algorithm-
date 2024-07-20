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

        self.actions = ['up', 'down', 'left', 'right']
        self.behavior_policy_probs = {a: 0.25 for a in self.actions}  # Equiprobable policy

        self.values = np.zeros((self.grid_size, self.grid_size))  # Value function, initializing all values of states to be 0
        self.policy = np.full((self.grid_size, self.grid_size), ' ')  # Policy array
        self.target_policy_probs = np.full((self.grid_size, self.grid_size, len(self.actions)), 1.0 / len(self.actions))  # Initialize target policy with equal probabilities

        self.highest_value_label = tk.Label(self.master, text="")
        self.highest_value_label.pack()

        # Dropdown menu to select evaluation method
        self.method_var = tk.StringVar(master)
        self.method_var.set("Monte Carlo with Importance Sampling")
        self.method_menu = tk.OptionMenu(master, self.method_var, "Monte Carlo with Importance Sampling")
        self.method_menu.pack()

        self.start_button = tk.Button(master, text="Start", command=self.start_evaluation)
        self.start_button.pack()

        self.draw_grid()

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

    def update_values_display(self):
        self.canvas.delete("values")
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                x0 = j * self.cell_size
                y0 = i * self.cell_size
                self.canvas.create_text(x0 + self.cell_size / 2, y0 + self.cell_size / 2,
                                        text=f"{self.values[i, j]:.2f}", tags="values")

    def update_behavioral_policy_display(self):
        self.canvas.delete("behavioral_policy")
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                x0 = j * self.cell_size
                y0 = i * self.cell_size
                self.canvas.create_text(x0 + self.cell_size / 2, y0 + self.cell_size - 20, text=self.policy[i, j],
                                        tags="behavioral_policy", fill="black")

    def update_target_policy_display(self):
        self.canvas.delete("target_policy")
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                x0 = j * self.cell_size
                y0 = i * self.cell_size
                self.canvas.create_text(x0 + self.cell_size / 2, y0 + self.cell_size - 20, text=self.policy[i, j],
                                        tags="target_policy", fill="black")

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

    def generate_episode(self):
        episode = []
        state = (0, 0)  # Start from a fixed initial state

        while state not in [(2, 4), (4, 0)]:
            action = random.choices(self.actions, weights=[self.behavior_policy_probs[a] for a in self.actions])[0]
            ni, nj, reward = self.get_next_state(*state, action)
            episode.append((state, action, reward))
            state = (ni, nj)

        return episode

    def Monte_Carlo_Importance_Sampling(self, gamma=0.95, epsilon=0.1, update_interval=100):
        print("Running Monte Carlo with Importance Sampling")
        self.returns = {}
        C = {}  # Cumulative weights

        for i in range(self.grid_size):
            for j in range(self.grid_size):
                self.returns[(i, j)] = {action: 0 for action in self.actions}
                C[(i, j)] = {action: 0 for action in self.actions}

        for episode_num in range(10000):  # Number of episodes
            episode = self.generate_episode()
            G = 0
            W = 1.0  # Importance sampling weight

            for t in reversed(range(len(episode))):
                # For each timestep t in the episode, extract state, action, reward
                state, action, reward = episode[t]
                G = gamma * G + reward
                C[state][action] += W
                self.returns[state][action] += W * (G - self.returns[state][action]) / C[state][action]
                self.values[state] += W * (G - self.values[state]) / C[state][action]

                # Update the target policy probability distribution
                best_action = max(self.actions, key=lambda a: self.returns[state][a])
                for a in self.actions:
                    self.target_policy_probs[state][self.actions.index(a)] = epsilon / len(self.actions)
                self.target_policy_probs[state][self.actions.index(best_action)] += (1.0 - epsilon)
                self.policy[state] = best_action  # Update the policy for this state

                # Compute importance weight for the next step
                W *= self.target_policy_probs[state][self.actions.index(action)] / self.behavior_policy_probs[action]

                if W == 0:
                    break

            # Update displays at intervals to avoid performance issues
            if (episode_num + 1) % update_interval == 0:
                self.update_values_display()
                self.update_behavioral_policy_display()
                self.master.update()

        # Final update after all episodes
        self.update_values_display()
        self.update_behavioral_policy_display()
        self.update_target_policy_display()

    def start_evaluation(self):
        self.highest_value_label.config(text="Running Monte Carlo with Importance Sampling")
        self.Monte_Carlo_Importance_Sampling()

if __name__ == "__main__":
    root = tk.Tk()
    grid_world = GridWorld(root)
    root.mainloop()
