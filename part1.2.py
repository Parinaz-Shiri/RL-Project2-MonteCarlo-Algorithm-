import tkinter as tk
import numpy as np

class GridWorld:
    def __init__(self, master):
        self.master = master
        self.master.title("GridWorld")

        self.grid_size = 5
        self.cell_size = 100

        canvas_height = self.grid_size * self.cell_size + 50
        self.canvas = tk.Canvas(master, width=self.grid_size * self.cell_size, height=canvas_height)
        self.canvas.pack()


        self.actions = ['up', 'down', 'left', 'right']
        self.action_probs = {'up': 0.25, 'down': 0.25, 'left': 0.25, 'right': 0.25}

        self.initialize_values()

        self.highest_value_label = tk.Label(self.master, text="")
        self.highest_value_label.pack()

        # Dropdown menu to select evaluation method
        self.method_var = tk.StringVar(master)
        self.method_var.set("Value Iteration")
        self.method_menu = tk.OptionMenu(master, self.method_var, "Bellman Equation", "Iterative Policy Evaluation", "Value Iteration")
        self.method_menu.pack()

        self.start_button = tk.Button(master, text="Start", command=self.start_evaluation)
        self.start_button.pack()

        self.reset_button = tk.Button(master, text="Reset", command=self.reset_values)
        self.reset_button.pack()

        self.draw_grid()
        self.update_values()

    def initialize_values(self):
        self.values = np.zeros((self.grid_size, self.grid_size))
        self.policy = np.full((self.grid_size, self.grid_size), "", dtype=object)

    def draw_grid(self):
        self.colors = {
            (0, 1): "blue",
            (0, 4): "green",
            (3, 2): "red",
            (4,4): "yellow"
        }

        for i in range(self.grid_size):
            for j in range(self.grid_size):
                x0 = j * self.cell_size
                y0 = i * self.cell_size
                x1 = x0 + self.cell_size
                y1 = y0 + self.cell_size
                color = self.colors.get((i, j), "white")
                self.canvas.create_rectangle(x0, y0, x1, y1, fill=color, outline="black")
                self.canvas.create_text(x0 + self.cell_size / 2, y0 + self.cell_size / 2,
                                        text=f"{self.values[i, j]:.2f}", tags="values")

    def update_values(self):
        self.canvas.delete("values")
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                x0 = j * self.cell_size
                y0 = i * self.cell_size

    def update_policy_display_old(self, policy):
        self.canvas.delete("policy")
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                x0 = j * self.cell_size
                y0 = i * self.cell_size
                action = policy[i, j]
                if action:
                    self.canvas.create_text(x0 + self.cell_size / 2, y0 + self.cell_size - 20, text=action,
                                            tags="policy", fill="black")

    def update_policy_display(self, policy): # To display the arrows
        self.canvas.delete("policy")
        
        arrow_offsets = {
            'U': (0, -0.5),
            'D': (0, 0.5),
            'L': (-0.5, 0),
            'R': (0.5, 0)
        }
        
        arrow_length = self.cell_size * 0.4
        arrow_head_width = 0.1 * arrow_length
        
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                x0 = j * self.cell_size + self.cell_size / 2
                y0 = i * self.cell_size + self.cell_size / 2
                actions = policy[i, j].split(', ')
                
                for idx, action in enumerate(actions):
                    if action:
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
        print("Running Bellman Iteration")
        while True:
            new_values = np.zeros((self.grid_size, self.grid_size))
            new_policy = np.full((self.grid_size, self.grid_size), "", dtype=object)
            for i in range(self.grid_size):
                for j in range(self.grid_size):
                    action_values = []
                    for action in self.actions:
                        ni, nj, reward = self.get_next_state(i, j, action)
                        action_values.append(reward + gamma * self.values[ni, nj])
                    max_value = max(action_values)
                    new_values[i, j] = max_value
                    best_actions = [self.actions[k] for k in range(len(action_values)) if action_values[k] == max_value]
                    new_policy[i, j] = ', '.join([a[0].upper() for a in best_actions])

            if np.sum(np.abs(new_values - self.values)) < epsilon:
                break
            self.values = new_values
            self.policy = new_policy
            self.update_values()
            self.master.update()
#
        self.update_policy_display(self.policy)
        self.display_highest_value_states()
        self.display_optimal_policy(self.policy)

    def Iterative_Policy_Evaluation_old(self, gamma=0.95, epsilon=0.01):
        print("Running Iterative Policy Evaluation")
        policy = np.full((self.grid_size, self.grid_size), "", dtype=object)
        while True:
            new_values = np.zeros((self.grid_size, self.grid_size))
            new_policy = np.full((self.grid_size, self.grid_size), "", dtype=object)
            for i in range(self.grid_size):
                for j in range(self.grid_size):
                    action_values = []
                    for action in self.actions:
                        ni, nj, reward = self.get_next_state(i, j, action)
                        action_values.append(reward + gamma * self.values[ni, nj])
                    max_value = max(action_values)
                    new_values[i, j] = max_value
                    best_actions = [self.actions[k] for k in range(len(action_values)) if action_values[k] == max_value]
                    new_policy[i, j] = ', '.join([a[0].upper() for a in best_actions])  # Show all optimal actions

            delta_matrix = np.abs(self.values - new_values)
            delta = np.max(delta_matrix)
            if delta < epsilon:
                break
            self.values = new_values
            policy = new_policy
            self.update_values()
            self.master.update()

        self.update_policy_display(policy)
        self.display_highest_value_states()
        self.display_optimal_policy(policy)
    
    def Iterative_Policy_Evaluation(self, gamma=0.95, theta=0.01):
        print("Running Iterative Policy Evaluation")

        while True:
            # Policy Evaluation
            while True:
                delta = 0
                for i in range(self.grid_size):
                    for j in range(self.grid_size):
                        v = self.values[i, j]
                        # Sum over possible actions for current policy
                        action_values = []
                        for action in self.actions:
                            ni, nj, reward = self.get_next_state(i, j, action)
                            action_values.append(reward + gamma * self.values[ni, nj])
                        new_value = np.max(action_values)  # Max value over all actions for the current policy
                        delta = max(delta, abs(v - new_value))
                        self.values[i, j] = new_value
                if delta < theta:
                    break

            # Policy Improvement
            policy_stable = True
            for i in range(self.grid_size):
                for j in range(self.grid_size):
                    old_action = self.policy[i, j]
                    action_values = []
                    for action in self.actions:
                        ni, nj, reward = self.get_next_state(i, j, action)
                        action_values.append(reward + gamma * self.values[ni, nj])
                    
                    max_value = max(action_values)
                    best_actions = [self.actions[k] for k in range(len(action_values)) if action_values[k] == max_value]
                    best_actions_str = ', '.join([a[0].upper() for a in best_actions])
                    self.policy[i, j] = best_actions_str
                    
                    if old_action != best_actions_str:
                        policy_stable = False

            self.update_values()
            self.update_policy_display(self.policy)
            self.master.update()

            if policy_stable:
                break

        self.display_highest_value_states()
        self.display_optimal_policy(self.policy)


        
    def Value_Iteration(self, gamma=0.95, epsilon=0.001):
        print("Running Value Iteration")
        
        while True:
            new_values = np.zeros((self.grid_size, self.grid_size))
            new_policy = np.full((self.grid_size, self.grid_size), "", dtype=object)
            delta = 0
            
            for i in range(self.grid_size):
                for j in range(self.grid_size):
                    action_values = []
                    for action in self.actions:
                        ni, nj, reward = self.get_next_state(i, j, action)
                        action_values.append(reward + gamma * self.values[ni, nj])
                    max_value = max(action_values)
                    new_values[i, j] = max_value
                    best_actions = [self.actions[k] for k in range(len(action_values)) if action_values[k] == max_value]
                    new_policy[i, j] = ', '.join([a[0].upper() for a in best_actions]) #prints only the first letter of the four actions
                    
                    delta = max(delta, abs(self.values[i, j] - new_values[i, j]))

            # Check for convergence
            if delta < epsilon:
                break
            
            self.values = new_values
            self.policy = new_policy
            self.update_values()
            self.master.update()

        self.update_policy_display(self.policy)
        self.display_highest_value_states()
        self.display_optimal_policy(self.policy)

    def start_evaluation(self):
        self.reset_values()
        method = self.method_var.get()
        self.highest_value_label.config(text=f"Running selected option: {method}")
        if method == "Bellman Equation":
            self.Bellman_Equation()
        elif method == "Iterative Policy Evaluation":
            self.Iterative_Policy_Evaluation()
        elif method == "Value Iteration":
            self.Value_Iteration()

    def reset_values(self):
        self.initialize_values()
        self.update_values()
        self.canvas.delete("policy")
        self.highest_value_label.config(text="")

    def display_highest_value_states(self):
        highest_value_states = np.argwhere(self.values == np.max(self.values))
        highest_value_text = f"States with the highest value: {highest_value_states.tolist()}, Value: {np.max(self.values):.2f}"
        self.highest_value_label.config(text=highest_value_text)

        print("States with the highest value:")
        for state in highest_value_states:
            # print(f"State: {state}, Value: {self.values[state[0], state[1]]}")
            print(f"State: {state}")


        print("\nFinal value function:")
        print(self.values)

    def display_optimal_policy(self, policy):
        print("Optimal Policy:")
        policy_display = ""
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                policy_display += f"{policy[i, j]} "
            policy_display += "\n"
        print(policy_display)


if __name__ == "__main__":
    root = tk.Tk()
    grid_world = GridWorld(root)
    root.mainloop()
