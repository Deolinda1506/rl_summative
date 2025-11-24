# training/train_a2c.py

import os
import csv
import numpy as np
from stable_baselines3 import A2C
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from training.CustomLogger import StrokeLoggerCallback
from environment.custom_env import StrokeDetectionEnv

# Directories
os.makedirs("models/a2c", exist_ok=True)
os.makedirs("logs", exist_ok=True)

CSV_PATH = "logs/a2c_hyperparams.csv"

# Create CSV if not exists
if not os.path.exists(CSV_PATH):
    with open(CSV_PATH, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "run_id", "learning_rate", "gamma", "n_steps", "ent_coef",
            "vf_coef", "max_grad_norm", "total_reward"
        ])

def make_env():
    return Monitor(StrokeDetectionEnv())

def safe_reset(env):
    """Handle both gym and gymnasium style reset returns."""
    out = env.reset()
    if isinstance(out, tuple):
        return out[0]
    return out

def safe_step(env, action):
    """Return obs, reward, done, info scalar-friendly."""
    out = env.step(action)
    if len(out) == 4:
        obs, reward, done, info = out
    elif len(out) == 5:
        obs, reward, terminated, truncated, info = out
        done = terminated or truncated
    else:
        raise ValueError("Unexpected env.step output length:", len(out))
    return obs, float(reward), bool(done), info

def evaluate_model_sb3(model, eval_env, episodes=5, max_steps=200):
    rewards = []
    for _ in range(episodes):
        obs = safe_reset(eval_env)
        total = 0.0
        for _ in range(max_steps):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, _ = safe_step(eval_env, action)
            total += reward
            if done:
                break
        rewards.append(total)
    return float(np.mean(rewards)), float(np.std(rewards))

# Hyperparameter configurations (10 runs)
configs = [
    {"learning_rate": 7e-4, "gamma": 0.99, "n_steps": 5, "ent_coef": 0.0, "vf_coef": 0.25, "max_grad_norm": 0.5},
    {"learning_rate": 3e-4, "gamma": 0.98, "n_steps": 20, "ent_coef": 0.01, "vf_coef": 0.5, "max_grad_norm": 0.5},
    {"learning_rate": 1e-3, "gamma": 0.95, "n_steps": 5, "ent_coef": 0.0, "vf_coef": 0.25, "max_grad_norm": 0.5},
    {"learning_rate": 5e-4, "gamma": 0.99, "n_steps": 10, "ent_coef": 0.001, "vf_coef": 0.25, "max_grad_norm": 0.5},
    {"learning_rate": 2.5e-4, "gamma": 0.97, "n_steps": 20, "ent_coef": 0.01, "vf_coef": 0.5, "max_grad_norm": 1.0},
    {"learning_rate": 7e-4, "gamma": 0.95, "n_steps": 10, "ent_coef": 0.0, "vf_coef": 0.1, "max_grad_norm": 0.5},
    {"learning_rate": 1e-4, "gamma": 0.99, "n_steps": 5, "ent_coef": 0.02, "vf_coef": 0.25, "max_grad_norm": 0.5},
    {"learning_rate": 3e-4, "gamma": 0.96, "n_steps": 20, "ent_coef": 0.0, "vf_coef": 0.5, "max_grad_norm": 1.0},
    {"learning_rate": 5e-4, "gamma": 0.98, "n_steps": 10, "ent_coef": 0.005, "vf_coef": 0.25, "max_grad_norm": 0.5},
    {"learning_rate": 3e-4, "gamma": 0.99, "n_steps": 20, "ent_coef": 0.0, "vf_coef": 0.25, "max_grad_norm": 0.5},
]

TIMESTEPS = 50000

for idx, cfg in enumerate(configs, start=1):
    run_name = f"a2c_run{idx}"
    print(f"\n=== Starting {run_name} ===\nParams: {cfg}\n")

    env = DummyVecEnv([make_env])
    callback = StrokeLoggerCallback(verbose=0)

    model = A2C(
        "MlpPolicy",
        env,
        verbose=1,
        learning_rate=cfg["learning_rate"],
        gamma=cfg["gamma"],
        n_steps=cfg["n_steps"],
        ent_coef=cfg["ent_coef"],
        vf_coef=cfg["vf_coef"],
        max_grad_norm=cfg["max_grad_norm"]
    )

    model.learn(total_timesteps=TIMESTEPS, callback=callback)

    # Save model
    save_dir = f"models/a2c/run_{idx}"
    os.makedirs(save_dir, exist_ok=True)
    model.save(f"{save_dir}/a2c_model")

    # Total reward
    total_reward = sum(callback.episode_rewards)

    # Log hyperparameters + reward
    with open(CSV_PATH, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            idx,
            cfg["learning_rate"],
            cfg["gamma"],
            cfg["n_steps"],
            cfg["ent_coef"],
            cfg["vf_coef"],
            cfg["max_grad_norm"],
            total_reward
        ])

    env.close()
    print(f"Finished {run_name}: total_reward={total_reward:.2f}\nSaved to {save_dir}/a2c_model\n")

print("All A2C runs done. Logs saved to logs/a2c_hyperparams.csv")
