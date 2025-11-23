# training/dqn_training.py

import os
import itertools
import pandas as pd
from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback
from environment.custom_env import StrokeDetectionEnv

# Simple callback for console logging
class StrokeLoggerCallback(BaseCallback):
    def __init__(self, verbose=1):
        super().__init__(verbose)

    def _on_step(self) -> bool:
        return True

# Create necessary directories
os.makedirs("models/dqn", exist_ok=True)
os.makedirs("logs/dqn", exist_ok=True)

# Define hyperparameter grid
learning_rates = [1e-3, 5e-4, 1e-4]
gammas = [0.9, 0.95, 0.99]
batch_sizes = [32, 64]
buffer_sizes = [20000, 50000]

# Track results
results = []

# Create environment function
def make_env():
    return DummyVecEnv([lambda: Monitor(StrokeDetectionEnv())])

# Hyperparameter tuning loop
for lr, gamma, batch_size, buffer_size in itertools.product(
    learning_rates, gammas, batch_sizes, buffer_sizes
):
    # Model name for saving
    model_name = f"dqn_lr{lr}_gamma{gamma}_bs{batch_size}_buf{buffer_size}"
    print(f"\n=== Training {model_name} ===")

    # Create environment
    env = make_env()

    # Create DQN model
    model = DQN(
        "MlpPolicy",
        env,
        verbose=1,
        learning_rate=lr,
        gamma=gamma,
        batch_size=batch_size,
        buffer_size=buffer_size,
        learning_starts=1000,
        exploration_fraction=0.1,
        exploration_final_eps=0.02
    )

    # Train model
    model.learn(total_timesteps=50000, callback=StrokeLoggerCallback())

    # Save model
    model_path = f"models/dqn/{model_name}.zip"
    model.save(model_path)
    print(f"Model saved at {model_path}")

    # Evaluate model
    obs = env.reset()  # obs is an array (batch_size=1)
    total_reward = 0
    for step in range(200):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        total_reward += reward[0]  # reward is array
        if done[0]:  # done is array
            break

    print(f"Total reward after evaluation: {total_reward:.2f}")

    # Record result
    results.append({
        "model": model_name,
        "learning_rate": lr,
        "gamma": gamma,
        "batch_size": batch_size,
        "buffer_size": buffer_size,
        "total_reward": float(total_reward)
    })

    env.close()

# Save all results to CSV for report
results_df = pd.DataFrame(results)
results_df.to_csv("logs/dqn/hyperparam_results.csv", index=False)
print("All hyperparameter tuning results saved at logs/dqn/hyperparam_results.csv")
