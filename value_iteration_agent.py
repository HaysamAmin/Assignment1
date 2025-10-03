#!/usr/bin/python3

import numpy as np

class Agent:
    def __init__(self, env, theta_threshold=0.01):
        """
        Agent for Value Iteration.
        """
        self.env = env
        self.env_size = env.get_size()
        self.theta_threshold = float(theta_threshold)

        # Value function V(s)
        self.V = np.zeros((self.env_size, self.env_size), dtype=float)
        if hasattr(self.env, "update_value_function"):
            self.env.update_value_function(self.V)

        # MDP pieces from env
        self.actions = env.get_actions()
        self.action_descriptions = env.get_action_descriptions()
        self.gamma = float(env.get_gamma())

        # Policy storage
        self.policy_grid = np.full((self.env_size, self.env_size), "", dtype=object)
        self.pi_greedy = np.full((self.env_size, self.env_size), -1, dtype=int)
        self.pi_str = []

    # ------------------ METHODS ------------------

    def calculate_max_value(self, i, j):
        """One-step Bellman update."""
        max_value = float('-inf')
        best_action = None
        best_actions_str = ""

        for action_index in range(len(self.actions)):
            next_i, next_j, reward, _ = self.env.step(action_index, i, j)
            if self.env.is_valid_state(next_i, next_j):
                value = self.get_value(next_i, next_j, reward)
                if value > max_value:
                    max_value = value
                    best_action = action_index
                    best_actions_str = self.action_descriptions[action_index]
                elif value == max_value:  # tie
                    best_actions_str += "|" + self.action_descriptions[action_index]

        return max_value, best_action, best_actions_str

    def get_value(self, i, j, reward):
        """Bellman backup for one state-action."""
        return reward + self.gamma * self.V[i, j]

    def update_value_function(self, V):
        """Replace V with new estimate."""
        self.V = np.copy(V)

    def get_value_function(self):
        """Return the current value function."""
        return self.V

    def update_greedy_policy(self):
        """Update greedy policy π(s)."""
        self.pi_str = []
        for i in range(self.env_size):
            pi_row = []
            for j in range(self.env_size):
                if self.env.is_terminal_state(i, j):
                    pi_row.append("X")
                    self.pi_greedy[i, j] = -1
                    continue
                _, self.pi_greedy[i, j], action_str = self.calculate_max_value(i, j)
                pi_row.append(action_str)
            self.pi_str.append(pi_row)

    def is_done(self, new_V):
        """Check convergence threshold."""
        delta = abs(self.V - new_V)
        return delta.max() <= self.theta_threshold

    def get_policy(self):
        """Return greedy policy indices."""
        return self.pi_greedy

    def print_policy(self):
        """Print stored policy in human-readable form."""
        for row in self.pi_str:
            print(row)
