#!/usr/bin/python3
import numpy as np

class GridWorld:
    """
    5x5 GridWorld environment for Value Iteration.

    - Terminal/goal: (4,4) with reward +10
    - Grey/bad states: (1,2), (3,0), (0,4) with reward -5
    - All other states: reward -1
    - Deterministic transitions; off-grid => stay in place
    """

    def __init__(self, env_size=5, gamma=0.95):
        self.env_size = env_size
        self.gamma = float(gamma)

        # --- Rewards (Problem 3 spec) ---
        self.goal = (env_size - 1, env_size - 1)           # (4,4) for 5x5
        self.greys = {(1, 2), (3, 0), (0, 4)}               # bad states
        self.R = -1 * np.ones((env_size, env_size), dtype=float)
        self.R[self.goal] = 10.0
        for g in self.greys:
            self.R[g] = -5.0

        # --- Value function V(s) ---
        self.V = np.zeros((env_size, env_size), dtype=float)
        # You may keep V(goal)=0 or R(goal); many setups use 0 for terminal baseline
        # self.V[self.goal] = self.R[self.goal]

        # --- Actions ---
        # order: Right, Left, Down, Up  (matches many templates)
        self.actions = [(0, +1), (0, -1), (+1, 0), (-1, 0)]
        self.action_description = ["Right", "Left", "Down", "Up"]

        # Greedy policy grid (optional; agent may keep its own)
        self.pi_greedy = np.full((env_size, env_size), -1, dtype=int)

    # ------------------------------------------------------------------
    # Accessors expected by Agent
    # ------------------------------------------------------------------
    def get_size(self):
        return self.env_size

    def get_actions(self):
        """Return list of action displacement vectors."""
        return self.actions

    def get_action_descriptions(self):
        return self.action_description

    def get_gamma(self):
        return self.gamma

    def get_reward(self, i, j):
        """State-based reward R(s)."""
        return float(self.R[i, j])

    def is_terminal_state(self, i, j):
        return (i, j) == self.goal

    def get_value_function(self):
        return self.V

    def update_value_function(self, V_new):
        self.V = np.array(V_new, dtype=float, copy=True)

    # ------------------------------------------------------------------
    # Dynamics helpers
    # ------------------------------------------------------------------
    def step(self, action_index, i, j):
        """
        Apply action at (i, j). Off-grid -> stay in place.
        Returns: (ni, nj, reward, done)
        - ni, nj: next state indices
        - reward: R(ni, nj)  (state-based reward after the move)
        - done:   True if next state is terminal (goal), else False
     """
        di, dj = self.actions[action_index]
        ni, nj = i + di, j + dj

    # bounce off borders: stay if off-grid
        if not (0 <= ni < self.env_size and 0 <= nj < self.env_size):
            ni, nj = i, j

        reward = self.get_reward(ni, nj)                # reward of the state we land in
        done = self.is_terminal_state(ni, nj)           # terminal if goal
        return ni, nj, reward, done


    def is_valid_state(self, i, j):
        return 0 <= i < self.env_size and 0 <= j < self.env_size

    # ------------------------------------------------------------------
    # Optional: environment-side Bellman one-step lookahead
    # (Agent can call its own version; this is provided for convenience.)
    # ------------------------------------------------------------------
    def calculate_max_value(self, i, j):
        """
        One-step Bellman optimality:
        V_new(s) = max_a [ R(s) + γ * V(s') ], deterministic transitions.
        Returns (best_value, best_action_index, best_action_str).
        """
        if self.is_terminal_state(i, j):
            # Convention choices:
            #   return terminal reward, or keep V(goal) baseline (often 0).
            # Here we return R(goal) to make the terminal attractive in the final step.
            return self.get_reward(i, j), -1, "Goal"

        best_val = -1e18
        best_idx = -1
        best_str = ""

        r_s = self.get_reward(i, j)
        gamma = self.get_gamma()

        for a_idx in range(len(self.actions)):
            ni, nj, _, _ = self.step(a_idx, i, j)
            q = r_s + gamma * self.V[ni, nj]
            if q > best_val:
                best_val = q
                best_idx = a_idx
                best_str = self.action_description[a_idx]

        return best_val, best_idx, best_str

    # ------------------------------------------------------------------
    # Optional: build a greedy policy grid from current V
    # ------------------------------------------------------------------
    def update_greedy_policy(self):
        for i in range(self.env_size):
            for j in range(self.env_size):
                if self.is_terminal_state(i, j):
                    self.pi_greedy[i, j] = -1
                else:
                    _, a_idx, _ = self.calculate_max_value(i, j)
                    self.pi_greedy[i, j] = a_idx

    def print_policy(self):
        arrow = {0: "→", 1: "←", 2: "↓", 3: "↑", -1: "G"}
        for i in range(self.env_size):
            print(" ".join(f"{arrow[self.pi_greedy[i,j]]:>2}" for j in range(self.env_size)))
