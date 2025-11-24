# training/train_ppo.py

import os
import csv
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from training.CustomLogger import StrokeLoggerCallback
from environment.custom_env import StrokeDetectionEnv

# Create directories
os.makedirs("models/pg", exist_ok=True)
os.makedirs("logs", exist_ok=True)

CSV_PATH = "logs/ppo_hyperparams.csv"

# Define 10 hyperparameter configurations
ppo_configs = [
    {"learning_rate": 0.0003, "gamma": 0.99, "n_steps": 1024, "batch_size": 32},
    {"learning_rate": 0.0001, "gamma": 0.98, "n_steps": 512, "batch_size": 64},
    {"learning_rate": 0.0005, "gamma": 0.95, "n_steps": 2048, "batch_size": 32},
    {"learning_rate": 0.0007, "gamma": 0.97, "n_steps": 1024, "batch_size": 128},
    {"learning_rate": 0.0003, "gamma": 0.99, "n_steps": 2048, "batch_size": 64},
    {"learning_rate": 0.0001, "gamma": 0.99, "n_steps": 4096, "batch_size": 32},
    {"learning_rate": 0.0002, "gamma": 0.96, "n_steps": 2048, "batch_size": 128},
    {"learning_rate": 0.0005, "gamma": 0.98, "n_steps": 512, "batch_size": 32},
    {"learning_rate": 0.0003, "gamma": 0.97, "n_steps": 1024, "batch_size": 64},
    {"learning_rate": 0.00005, "gamma": 0.99, "n_steps": 4096, "batch_size": 16},
]

# Create CSV if not exists
if not os.path.exists(CSV_PATH):
    with open(CSV_PATH, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["run_id", "learning_rate", "gamma", "n_steps", "batch_size", "total_reward"])

# Training loop for 10 variants
for i, cfg in enumerate(ppo_configs):
    run_id = i + 1
    print(f"\n Starting PPO Training Run {run_id}/10 with params: {cfg}")

    # Environment
    env = DummyVecEnv([lambda: Monitor(StrokeDetectionEnv())])
    logger_callback = StrokeLoggerCallback(verbose=0)

    # Create model
    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        learning_rate=cfg["learning_rate"],
        gamma=cfg["gamma"],
        n_steps=cfg["n_steps"],
        batch_size=cfg["batch_size"]
    )

    # Train
    model.learn(total_timesteps=50000, callback=logger_callback)

    # Save model
    save_path = f"models/pg/run_{run_id}"
    os.makedirs(save_path, exist_ok=True)
    model.save(f"{save_path}/ppo_model")

    # Retrieve total reward from logger
    total_reward = sum(logger_callback.episode_rewards)  # fixed

    # Log hyperparameters + reward
    with open(CSV_PATH, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            run_id,
            cfg["learning_rate"],
            cfg["gamma"],
            cfg["n_steps"],
            cfg["batch_size"],
            total_reward
        ])

    env.close()

    print(f" Run {run_id} complete. Model saved to {save_path}/ppo_model")

print("\n PPO hyperparameter search completed! Logs saved to logs/ppo_hyperparams.csv")
