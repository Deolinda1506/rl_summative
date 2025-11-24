# main.py
import os
import pandas as pd
import torch
import numpy as np
from stable_baselines3 import DQN, PPO, A2C
from gymnasium.wrappers import RecordVideo
from environment.custom_env import StrokeDetectionEnv

# -------------------------------
# Function to pick the best run
# -------------------------------
def pick_best_run(csv_path, reward_col):
    df = pd.read_csv(csv_path)
    if reward_col not in df.columns:
        raise ValueError(f"No reward column '{reward_col}' in {csv_path}")
    best_row = df.loc[df[reward_col].idxmax()]
    return best_row

# -------------------------------
# PolicyNet for REINFORCE
# -------------------------------
class PolicyNet(torch.nn.Module):
    def __init__(self, obs_size=5, n_actions=6, hidden=128):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(obs_size, hidden),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden, hidden),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden, n_actions)
        )

    def forward(self, x):
        return self.net(x)

# -------------------------------
# Safe env helpers
# -------------------------------
def safe_reset(env):
    out = env.reset()
    return out[0] if isinstance(out, tuple) else out

def safe_step(env, action):
    out = env.step(action)
    if len(out) == 4:
        obs, reward, done, info = out
    elif len(out) == 5:
        obs, reward, terminated, truncated, info = out
        done = terminated or truncated
    else:
        raise ValueError("Unexpected env.step output length")
    return obs, float(reward), bool(done), info

# -------------------------------
# Evaluate & record
# -------------------------------
def evaluate_and_record(model, env, episodes=3, model_name="model", reinforce_policy=False, device="cpu"):
    for episode in range(episodes):
        obs = safe_reset(env)
        total_reward = 0
        done = False
        step_count = 0

        while not done and step_count < 200:
            if reinforce_policy:
                with torch.no_grad():
                    logits = model(torch.tensor(obs, dtype=torch.float32, device=device))
                    probs = torch.softmax(logits, dim=-1).cpu().numpy()
                    action = int(np.argmax(probs))
            else:
                action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, _ = safe_step(env, action)
            total_reward += reward
            step_count += 1

        print(f"{model_name} Episode {episode+1}: Total Reward = {total_reward:.2f}")

    env.close()

# -------------------------------
# Load best models
# -------------------------------
best_dqn = pick_best_run("logs/dqn_hyperparams.csv", "total_reward")
best_ppo = pick_best_run("logs/ppo_hyperparams.csv", "total_reward")
best_a2c = pick_best_run("logs/a2c_hyperparams.csv", "total_reward")
best_reinforce = pick_best_run("logs/reinforce_hyperparams.csv", "mean_reward")

# Paths to models
model_dqn = DQN.load(f"models/dqn/run_{int(best_dqn['run_id'])}/dqn_model")
model_ppo = PPO.load(f"models/pg/run_{int(best_ppo['run_id'])}/ppo_model")
model_a2c = A2C.load(f"models/a2c/run_{int(best_a2c['run_id'])}/a2c_model")

# REINFORCE
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
env_tmp = StrokeDetectionEnv()
obs_size = env_tmp.observation_space.shape[0]
n_actions = env_tmp.action_space.n
policy_reinforce = PolicyNet(obs_size=obs_size, n_actions=n_actions, hidden=int(best_reinforce['hidden_size'])).to(device)
policy_reinforce.load_state_dict(torch.load(f"models/reinforce/run_{int(best_reinforce['run_name'])}/policy_state.pt", map_location=device))

# -------------------------------
# Create environments with video
# -------------------------------
env_dqn = RecordVideo(StrokeDetectionEnv(render_mode='rgb_array'), video_folder='rl_agent_videos_dqn', episode_trigger=lambda x: True)
env_ppo = RecordVideo(StrokeDetectionEnv(render_mode='rgb_array'), video_folder='rl_agent_videos_ppo', episode_trigger=lambda x: True)
env_a2c = RecordVideo(StrokeDetectionEnv(render_mode='rgb_array'), video_folder='rl_agent_videos_a2c', episode_trigger=lambda x: True)
env_reinforce = RecordVideo(StrokeDetectionEnv(render_mode='rgb_array'), video_folder='rl_agent_videos_reinforce', episode_trigger=lambda x: True)

# -------------------------------
# Evaluate all models
# -------------------------------
evaluate_and_record(model_dqn, env_dqn, model_name='DQN')
evaluate_and_record(model_ppo, env_ppo, model_name='PPO')
evaluate_and_record(model_a2c, env_a2c, model_name='A2C')
evaluate_and_record(policy_reinforce, env_reinforce, model_name='REINFORCE', reinforce_policy=True, device=device)
