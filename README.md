# Stroke Detection and Care System Using Reinforcement Learning and Computer Vision

Project Overview
-----------------
This project implements a stroke detection and patient monitoring system using reinforcement learning (RL) and simple computer-vision-style rendering. A drone agent monitors a patient on a 10x10 grid and attempts to detect stroke events promptly. The repository contains training code, evaluation, visualization utilities, and scripts to compare RL methods.

Image
-----
*(Add a representative image or link to your simulation GIF here)*

Links
-----
- Report: (add link)
- Video - Simulation: (add link)

Algorithms Compared
-------------------
- Deep Q-Network (DQN) — value-based method
- Proximal Policy Optimization (PPO) — policy-gradient method

Note: the repo also contains other training scripts (A2C, REINFORCE) for additional experiments, but the primary comparison and discussion here focus on DQN vs PPO.

Environment Description
-----------------------
- Grid size: 10x10 cells
- Drone (agent): moves horizontally and vertically; also supports zoom actions in the simulation
- Patient: randomly moves around the grid and sometimes experiences stroke events
- Stroke events: randomly triggered (configurable probability) and must be detected quickly by the agent
- Observation: 5-dimensional vector: `[drone_x, drone_y, zoom, patient_x, patient_y]`
- Action space: `Discrete(6)` — left, right, up, down, zoom_in, zoom_out
- Episode length: episodes end after 200 timesteps

Project Structure
-----------------
```
project_root/
├── environment/
│   ├── custom_env.py            # Custom Gymnasium environment
│   └── rendering.py             # Visualization utilities (pygame + imageio)
├── training/
│   ├── dqn_training.py          # Training script for DQN agent
│   ├── pg_training.py           # Training script for PPO agent (named pg_training.py)
│   ├── train_a2c.py             # A2C training script
│   ├── train_reinforce.py       # REINFORCE (PyTorch) training script
│   └── CustomLogger.py          # Callback for SB3 logging
├── models/                      # Saved model runs per algorithm
│   ├── dqn/
│   ├── pg/                      # PPO models (pg used in this repo)
│   └── reinforce/
├── main.py                      # Evaluate best models and record videos
├── requirements.txt             # Python dependencies
└── README.md                    # Project documentation (this file)
```

Step-by-step Summary & Findings
-------------------------------
Step 1 — Installation
- Set up a Python virtual environment and install dependencies from `requirements.txt` (see Quick Start below).

Step 2 — Training Models
- DQN (examples used in experiments):
	- learning rates tried: `1e-4`, `5e-4`, etc.
	- batch sizes and replay buffer sizes varied (e.g., batch=32, buffer=20000)
	- observed: effective exploration but slower convergence and occasional instability

- PPO (examples used in experiments):
	- learning rates tried: `3e-4`, `1e-4`, etc.
	- `n_steps` varied (e.g., 1024, 2048, 4096)
	- observed: faster convergence and more stable reward improvement compared to DQN

Step 3 — Evaluation & Recording Videos
- Use `python main.py` to select the best hyperparameter runs from `logs/*_hyperparams.csv`, load those models from `models/`, and record evaluation episodes into `rl_agent_videos_*` folders.
- In recorded videos, PPO runs typically show smoother and quicker detection of seizure events than DQN runs.

Step 4 — Visualization
- Run `environment/rendering.py` as a script to generate example GIFs or visualizations (`visualize_environment()` saves a GIF via `imageio`).

Performance Metrics & Hyperparameter Impact
-----------------------------------------
- PPO generally converged faster and attained higher, more stable rewards than DQN in our experiments.
- Hyperparameters that matter most:
	- learning rate: lower values improved stability (smaller, steadier updates)
	- gamma (discount factor): values around `0.98` balanced short-term detection and long-term planning well
	- buffer size and batch size (DQN): influenced stability and sample efficiency

Results & Comparison (summary)
------------------------------
- DQN: good at exploration, sometimes slower to converge and prone to instability without careful tuning.
- PPO: more stable, faster learning in this environment, and consistently higher average reward across runs.

Conclusion
----------
PPO outperformed DQN in this simulated epilepsy-detection task due to better stability and sample efficiency in the tested hyperparameter ranges. Hyperparameter tuning (learning rate, gamma, buffer/batch sizes) is critical for reproducible performance.

Future Improvements
-------------------
- Integrate real-time clinical data or synthetic video streams for more realistic observation inputs
- Model more realistic drone physics and sensor noise
- Expand experiments to actor-critic variants, distributional RL, and model-based approaches
- Add evaluation metrics beyond episodic reward (e.g., detection latency, false-positive rate)

Quick Start
-----------
1. Create and activate a virtual environment:

```bash
python3 -m venv venv
source venv/bin/activate
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Train (examples):

```bash
python training/dqn_training.py
python training/pg_training.py
```

4. Evaluate and record videos:

```bash
python main.py
```

Notes
-----
- `main.py` chooses best runs by reading `logs/*_hyperparams.csv`. Ensure those CSV files and model folders exist before running evaluation.
- REINFORCE models are stored as PyTorch state dicts (e.g., `models/reinforce/run_<i>/policy_state.pt`). SB3 saves full models (e.g., `models/dqn/run_<i>/dqn_model`).

If you'd like, I can:
- Add the image and video links into this README.
- Create a short `scripts/run_quick_demo.sh` that performs a very short training (reduced timesteps/episodes) and then runs evaluation to produce example outputs.
- Run a smoke-check here: install dependencies and run a short training + evaluation to verify everything works in the devcontainer.

---

Credits
-------
This project was developed as a research/educational summative project exploring RL for early medical event detection.


