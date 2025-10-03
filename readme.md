# Assignment 2 — Value Iteration & Off-Policy MC (Importance Sampling)

This repo contains solutions for a 5×5 GridWorld using:
- **Value Iteration (VI)** — synchronous and **in-place** variants
- **Off-Policy Monte Carlo (MC)** prediction with **Importance Sampling (IS)**

Outputs include a table/heatmap of the optimal value function **V\*** and a grid of arrows for the greedy policy **π\***, plus a short comparison of VI vs. MC-IS (time, sweeps/episodes, and complexity notes).

## Repository Contents
- `Part1-2.ipynb` — GridWorld setup, **Synchronous VI**, **In-Place VI**, parity check, figures.
- `Part3-4.ipynb` — **Off-Policy MC with IS**, timing & episode count, RMSE vs VI, summary.
- `value_iteration.py`, `value_iteration_agent.py`, `value_iteration_solved.py` — helper/driver code.
- `.gitignore` — ignores for Python/Jupyter projects.

## Quick Start
```bash
# 1) (Optional) create a virtual environment
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

# 2) Install deps (if no requirements.txt, install common stack)
pip install numpy matplotlib pandas jupyter

# 3) Launch Jupyter
jupyter lab   # or: jupyter notebook
