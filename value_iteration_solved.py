"""
Value Iteration Driver Script for 5x5 GridWorld

This script:
1. Creates a GridWorld environment of size 5x5.
2. Runs Value Iteration to compute the optimal state-value function V*.
3. Extracts the greedy optimal policy π*.
4. Prints both V* and π* for analysis.

Make sure:
- gridworld.py implements the GridWorld environment with correct rewards:
    +10 for the goal at (4,4)
    -5 for grey states {(1,2), (3,0), (0,4)}
    -1 for all other states
- value_iteration_agent.py implements the Agent class with:
    * Bellman update in calculate_max_value()
    * Convergence check in is_done()
    * Greedy policy extraction in update_greedy_policy()
"""

import numpy as np
from gridworld import GridWorld
from value_iteration_agent import Agent

def main():
    # -----------------------------
    # Parameters
    # -----------------------------
    ENV_SIZE = 5               # 5x5 grid
    THETA_THRESHOLD = 0.05     # Convergence tolerance: stop when Δ < theta
    MAX_ITERATIONS = 1000      # Safety cap on iterations

    # -----------------------------
    # Environment and Agent setup
    # -----------------------------
    env = GridWorld(ENV_SIZE)                # Create grid environment
    agent = Agent(env, THETA_THRESHOLD)      # Create agent with Bellman updates

    # -----------------------------
    # Value Iteration Loop
    # -----------------------------
    done = False
    for iter in range(MAX_ITERATIONS):
        # Stop if convergence has already been reached
        if done:
            break

        # Make a copy of the current value function (V_old)
        # This ensures synchronous updates: we don't overwrite V in the middle of a sweep
        new_V = np.copy(agent.get_value_function())

        # Sweep over every state in the grid
        for i in range(ENV_SIZE):
            for j in range(ENV_SIZE):
                # Skip terminal states (goal)
                if not env.is_terminal_state(i, j):
                    # Bellman optimality update:
                    # V_new(s) = max_a [ R(s) + γ * V_old(s') ]
                    new_V[i, j], _, _ = agent.calculate_max_value(i, j)

        # Check for convergence (Δ < theta)
        # Δ = max |V_new(s) - V_old(s)|
        done = agent.is_done(new_V)

        # Update the agent's value function
        agent.update_value_function(new_V)

    # -----------------------------
    # Print Results
    # -----------------------------
    print("Optimal Value Function Found in %d iterations:" % (iter + 1))
    print(agent.get_value_function())

    # Derive and print the greedy policy π* from V*
    agent.update_greedy_policy()
    agent.print_policy()

if __name__ == "__main__":
    main()
