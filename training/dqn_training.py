# training/dqn_training.py

import os
import csv
from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from training.CustomLogger import StrokeLoggerCallback
from environment.custom_env import StrokeDetectionEnv

# Create directories
os.makedirs("models/dqn", exist_ok=True)
os.makedirs("logs", exist_ok=True)

CSV_PATH = "logs/dqn_hyperparams.csv"

# Create CSV if not exists
if not os.path.exists(CSV_PATH):
    with open(CSV_PATH, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["run_id", "learning_rate", "gamma", "batch_size", "buffer_size",
                         "learning_starts", "exploration_fraction", "exploration_final_eps",
                         "tau", "train_freq", "total_reward"])

def make_env():
    return Monitor(StrokeDetectionEnv())

def train_dqn(run_id, params, timestep_limit=50000):
    print(f"\n Training DQN Run {run_id} with params: {params}\n")

    env = DummyVecEnv([make_env])
    callback = StrokeLoggerCallback(verbose=0)

    # Initialize DQN model
    model = DQN(
        "MlpPolicy",
        env,
        verbose=1,
        **params
    )

    # Train model
    model.learn(total_timesteps=timestep_limit, callback=callback)

    # Save model
    save_path = f"models/dqn/run_{run_id}"
    os.makedirs(save_path, exist_ok=True)
    model.save(f"{save_path}/dqn_model")

    # Calculate total reward
    total_reward = sum(callback.episode_rewards)

    # Log hyperparameters + reward to CSV
    with open(CSV_PATH, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            run_id,
            params.get("learning_rate"),
            params.get("gamma"),
            params.get("batch_size"),
            params.get("buffer_size"),
            params.get("learning_starts"),
            params.get("exploration_fraction"),
            params.get("exploration_final_eps"),
            params.get("tau", ""),
            params.get("train_freq", ""),
            total_reward
        ])

    env.close()
    print(f" Run {run_id} complete. Model saved to {save_path}/dqn_model")
    print(f" Hyperparameters logged to {CSV_PATH}\n")


# 10 DQN Hyperparameter Runs
experiments = {
    "dqn_run1": {"learning_rate": 0.0001, "gamma": 0.99, "batch_size": 64,
                 "buffer_size": 20000, "learning_starts": 1000,
                 "exploration_fraction": 0.1, "exploration_final_eps": 0.02},
    "dqn_run2": {"learning_rate": 0.0005, "gamma": 0.95, "batch_size": 32,
                 "buffer_size": 50000, "learning_starts": 500,
                 "exploration_fraction": 0.15, "exploration_final_eps": 0.05},
    "dqn_run3": {"learning_rate": 0.00005, "gamma": 0.98, "batch_size": 128,
                 "buffer_size": 30000, "learning_starts": 2000,
                 "exploration_fraction": 0.2, "exploration_final_eps": 0.01},
    "dqn_run4": {"learning_rate": 0.0002, "gamma": 0.90, "batch_size": 256,
                 "buffer_size": 100000, "learning_starts": 1000,
                 "exploration_fraction": 0.25, "exploration_final_eps": 0.02},
    "dqn_run5": {"learning_rate": 0.001, "gamma": 0.99, "batch_size": 64,
                 "buffer_size": 40000, "learning_starts": 1500,
                 "exploration_fraction": 0.12, "exploration_final_eps": 0.03},
    "dqn_run6": {"learning_rate": 0.0003, "gamma": 0.92, "batch_size": 32,
                 "buffer_size": 80000, "learning_starts": 2000,
                 "exploration_fraction": 0.18, "exploration_final_eps": 0.02},
    "dqn_run7": {"learning_rate": 0.00005, "gamma": 0.999, "batch_size": 256,
                 "buffer_size": 120000, "learning_starts": 500,
                 "exploration_fraction": 0.3, "exploration_final_eps": 0.1},
    "dqn_run8": {"learning_rate": 0.0004, "gamma": 0.97, "batch_size": 128,
                 "buffer_size": 60000, "learning_starts": 2000,
                 "exploration_fraction": 0.05, "exploration_final_eps": 0.01},
    "dqn_run9": {"learning_rate": 0.00015, "gamma": 0.94, "batch_size": 32,
                 "buffer_size": 150000, "learning_starts": 3000,
                 "exploration_fraction": 0.35, "exploration_final_eps": 0.05},
    "dqn_run10": {"learning_rate": 0.0001, "gamma": 0.99, "batch_size": 64,
                  "buffer_size": 20000, "learning_starts": 1000,
                  "exploration_fraction": 0.05, "exploration_final_eps": 0.005,
                  "tau": 0.005, "train_freq": 4}
}

# Run all experiments
for idx, (name, params) in enumerate(experiments.items(), start=1):
    train_dqn(idx, params)

print("\n All 10 DQN experiments completed! Logs saved to logs/dqn_hyperparams.csv\n")
