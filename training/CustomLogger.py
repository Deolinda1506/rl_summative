# training/logger.py

import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import DQN, PPO, A2C
from environment.custom_env import StrokeDetectionEnv
import os

# Ensure plots folder exists
os.makedirs("plots", exist_ok=True)

# Load trained models
model_dqn = DQN.load("models/dqn/stroke_dqn_model")
model_ppo = PPO.load("models/pg/stroke_ppo_model")
# model_a2c = A2C.load("models/pg/stroke_a2c_model")  # Uncomment if you train A2C

# Evaluation parameters
EPISODES = 5
MAX_STEPS = 200

def evaluate_model_cumulative(model, model_name):
    """
    Evaluate a trained model and return cumulative rewards per episode.
    """
    cumulative_rewards_all = []

    for ep in range(EPISODES):
        env = StrokeDetectionEnv()
        obs, _ = env.reset()
        total_reward = 0
        cumulative_rewards = []

        for step in range(MAX_STEPS):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            cumulative_rewards.append(total_reward)

            if terminated or truncated:
                break

        cumulative_rewards_all.append(cumulative_rewards)
        env.close()
    return cumulative_rewards_all

def plot_cumulative_rewards(models_dict):
    """
    Generate and save cumulative reward plots for all models.
    """
    plt.figure(figsize=(10,5))

    colors = ["orange", "blue", "green"]
    for idx, (name, rewards_list) in enumerate(models_dict.items()):
        for ep_rewards in rewards_list:
            plt.plot(ep_rewards, color=colors[idx], alpha=0.5)
    
    plt.xlabel("Step")
    plt.ylabel("Cumulative Reward")
    plt.title("Cumulative Reward per Step")
    plt.grid()
    plt.legend(list(models_dict.keys()))
    plt.savefig("plots/cumulative_rewards.png")
    plt.show()
    print("Cumulative reward plot saved as 'plots/cumulative_rewards.png'")

if __name__ == "__main__":
    models = {
        "DQN": evaluate_model_cumulative(model_dqn, "DQN"),
        "PPO": evaluate_model_cumulative(model_ppo, "PPO"),
        # "A2C": evaluate_model_cumulative(model_a2c, "A2C")  # Uncomment if using A2C
    }

    plot_cumulative_rewards(models)
