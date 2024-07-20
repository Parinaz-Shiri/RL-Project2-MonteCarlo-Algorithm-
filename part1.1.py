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

        self.actions = ['up', 'down', 'left', 'right']  # Equal probabilities for every action
        self.action_probs = {'up': 0.25, 'down': 0.25, 'left': 0.25, 'right': 0.25}

        self.values = np.zeros((self.grid_size, self.grid_size))  # Value function, initializing all values of states to be 0

        self.highest_value_label = tk.Label(self.master, text="")
        self.highest_value_label.pack()

        # Dropdown menu to select evaluation method
        self.method_var = tk.StringVar(master)
        self.method_var.set("Iterative Policy Evaluation")
        self.method_menu = tk.OptionMenu(master, self.method_var, "Bellman Equation", "Iterative Policy Evaluation", "Value Iteration")
        self.method_menu.pack()

        self.start_button = tk.Button(master, text="Start", command=self.start_evaluation)
        self.start_button.pack()

        self.draw_grid()
        self.update_values()

    def draw_grid(self):  # colors for special states
        self.colors = {
            (0, 1): "blue",
            (0, 4): "green",
            (3, 2): "red",
            (4, 4): "yellow"
        }

        for i in range(self.grid_size):
            for j in range(self.grid_size):
                x0 = j * self.cell_size
                y0 = i * self.cell_size
                x1 = x0 + self.cell_size
                y1 = y0 + self.cell_size
                color = self.colors.get((i, j), "white")
                self.canvas.create_rectangle(x0, y0, x1, y1, fill=color, outline="black")
                self.canvas.create_text(x0 + self.cell_size / 2, y0 + self.cell_size / 2, text=f"{self.values[i, j]:.2f}", tags="values")

    def update_values(self):
        self.canvas.delete("values")
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                x0 = j * self.cell_size
                y0 = i * self.cell_size
                self.canvas.create_text(x0 + self.cell_size / 2, y0 + self.cell_size / 2, text=f"{self.values[i, j]:.2f}", tags="values")

    def update_policy_display(self, policy):
        self.canvas.delete("policy")
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                x0 = j * self.cell_size
                y0 = i * self.cell_size
                action = policy[i, j]
                if action:
                    self.canvas.create_text(x0 + self.cell_size / 2, y0 + self.cell_size - 20, text=action, tags="policy", fill="black")

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

        # Agent moving outside grid
        if (i, j) == (0, 1):
            return 3, 2, 5.0  # Special state
        elif (i, j) == (0, 4):
            if np.random.rand() < 0.5:
                return 3, 2, 2.5
            else:
                return 4, 4, 2.5
        elif ni < 0 or ni >= self.grid_size or nj < 0 or nj >= self.grid_size:
            return i, j, -0.5  # Stay in the same position, negative reward
        else:
            return ni, nj, 0.0  # Normal state, zero reward

    def Bellman_Equation(self, gamma=0.95, epsilon=0.01):
        print("Running Bellman Equation")
        while True:
            new_values = np.zeros((self.grid_size, self.grid_size))
            for i in range(self.grid_size):
                for j in range(self.grid_size):
                    for action in self.actions:
                        ni, nj, reward = self.get_next_state(i, j, action)
                        new_values[i, j] += self.action_probs[action] * (reward + gamma * self.values[ni, nj])

            if np.sum(np.abs(new_values - self.values)) < epsilon:
                break
            self.values = new_values
            self.update_values()
            self.master.update()

        self.display_highest_value_states()

    def Iterative_Policy_Evaluation(self, gamma=0.95, epsilon=0.01):
        print("Running Iterative Policy Evaluation")
        while True:
            new_values = np.zeros((self.grid_size, self.grid_size))
            for i in range(self.grid_size):
                for j in range(self.grid_size):
                    for action in self.actions:
                        ni, nj, reward = self.get_next_state(i, j, action)
                        new_values[i, j] += self.action_probs[action] * (reward + gamma * self.values[ni, nj])

            delta_matrix = np.abs(self.values - new_values)
            delta = np.max(delta_matrix)
            if delta < epsilon:
                break
            self.values = new_values
            self.update_values()
            self.master.update()

        self.display_highest_value_states()

    def Value_Iteration(self, gamma=0.95, epsilon=0.01):
        print("Running Value Iteration")
        policy = np.full((self.grid_size, self.grid_size), "", dtype=object)
        while True:
            new_values = np.zeros((self.grid_size, self.grid_size))
            for i in range(self.grid_size):
                for j in range(self.grid_size):
                    action_values = []
                    for action in self.actions:
                        ni, nj, reward = self.get_next_state(i, j, action)
                        action_values.append(reward + gamma * self.values[ni, nj])
                    # new_values[i, j] = max(action_values)
                    max_value = max(action_values)
                    new_values[i, j] = max_value
                    best_actions = [self.actions[k] for k, v in enumerate(action_values) if v == max_value]
                    policy[i, j] = best_actions        

            if np.sum(np.abs(new_values - self.values)) < epsilon:
                break
            self.values = new_values
            self.update_values()
            self.master.update()

        # self.update_policy_display(policy)
        self.display_highest_value_states()

    def start_evaluation(self):
        method = self.method_var.get()
        self.highest_value_label.config(text=f"Running selected option: {method}")
        if method == "Bellman Equation":
            self.Bellman_Equation()
        elif method == "Iterative Policy Evaluation":
            self.Iterative_Policy_Evaluation()
        elif method == "Value Iteration":
            self.Value_Iteration()

    def display_highest_value_states(self):
        highest_value_states = np.argwhere(self.values == np.max(self.values))
        highest_value_text = f"States with the highest value: {highest_value_states.tolist()}, Value: {np.max(self.values):.2f}"
        self.highest_value_label.config(text=highest_value_text)

        print("States with the highest value:")
        for state in highest_value_states:
            print(f"State: {state}, Value: {self.values[state[0], state[1]]}")

        print("\nFinal value function:")
        print(self.values)

if __name__ == "__main__":
    root = tk.Tk()
    grid_world = GridWorld(root)
    root.mainloop()
