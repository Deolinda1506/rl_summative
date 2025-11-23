# training/pg_training.py

import os
import itertools
import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback
from environment.custom_env import StrokeDetectionEnv

# Callback for console logging
class StrokeLoggerCallback(BaseCallback):
    def __init__(self, verbose=1):
        super().__init__(verbose)

    def _on_step(self) -> bool:
        return True

# Create directories
os.makedirs("models/pg", exist_ok=True)
os.makedirs("logs/pg", exist_ok=True)

# Hyperparameter grid
learning_rates = [1e-4, 3e-4, 5e-4]
gammas = [0.9, 0.95, 0.99]
n_steps_list = [512, 1024, 2048]
batch_sizes = [32, 64]

# Store results
results = []

# Environment factory
def make_env():
    return DummyVecEnv([lambda: Monitor(StrokeDetectionEnv())])

# Hyperparameter tuning loop
for lr, gamma, n_steps, batch_size in itertools.product(
    learning_rates, gammas, n_steps_list, batch_sizes
):
    model_name = f"ppo_lr{lr}_gamma{gamma}_n{n_steps}_bs{batch_size}"
    print(f"\n=== Training {model_name} ===")

    # Create environment
    env = make_env()
    
    # Create PPO model
    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        learning_rate=lr,
        gamma=gamma,
        n_steps=n_steps,
        batch_size=batch_size
    )

    # Train model
    model.learn(total_timesteps=50000, callback=StrokeLoggerCallback())
    
    # Save model
    model_path = f"models/pg/{model_name}.zip"
    model.save(model_path)
    print(f"Model saved at {model_path}")

    # Evaluate model
    obs = env.reset()
    total_reward = 0
    for step in range(200):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)  # DummyVecEnv returns 4 items
        total_reward += reward[0]  # reward is an array
        if done[0]:
            break

    print(f"Total reward after evaluation: {total_reward:.2f}")

    # Record results
    results.append({
        "model": model_name,
        "learning_rate": lr,
        "gamma": gamma,
        "n_steps": n_steps,
        "batch_size": batch_size,
        "total_reward": total_reward
    })

    env.close()

# Save all results to CSV
results_df = pd.DataFrame(results)
results_df.to_csv("logs/pg/hyperparam_results.csv", index=False)
print("All hyperparameter tuning results saved at logs/pg/hyperparam_results.csv")
