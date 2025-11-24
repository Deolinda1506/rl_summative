# training/train_reinforce.py
import csv
import os
import math
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from environment.custom_env import StrokeDetectionEnv

os.makedirs("models/reinforce", exist_ok=True)
os.makedirs("logs", exist_ok=True)

CSV_PATH = "logs/reinforce_hyperparams.csv"

# Create CSV if not exists
if not os.path.exists(CSV_PATH):
    with open(CSV_PATH, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "run_id", "learning_rate", "gamma", "hidden_size",
            "episodes", "timesteps_done", "mean_reward"
        ])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Simple Policy Network
class PolicyNet(nn.Module):
    def __init__(self, obs_size=5, n_actions=6, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_size, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, n_actions)
        )

    def forward(self, x):
        return self.net(x)

def safe_reset(env):
    out = env.reset()
    if isinstance(out, tuple):
        return out[0]
    return out

def safe_step(env, action):
    out = env.step(action)
    if len(out) == 4:
        obs, reward, done, info = out
    elif len(out) == 5:
        obs, reward, terminated, truncated, info = out
        done = terminated or truncated
    else:
        raise ValueError("Unexpected step output length")
    return obs, float(reward), bool(done), info

def compute_returns(rewards, gamma):
    R = 0.0
    returns = []
    for r in reversed(rewards):
        R = r + gamma * R
        returns.insert(0, R)
    returns = torch.tensor(returns, dtype=torch.float32, device=device)
    # normalize
    if returns.std().item() > 1e-8:
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)
    return returns

def evaluate_policy(policy, env, episodes=5, max_steps=200):
    totals = []
    for _ in range(episodes):
        obs = safe_reset(env)
        total = 0.0
        for _ in range(max_steps):
            with torch.no_grad():
                logits = policy(torch.tensor(obs, dtype=torch.float32, device=device))
                probs = torch.softmax(logits, dim=-1).cpu().numpy()
            a = int(np.argmax(probs))
            obs, r, done, _ = safe_step(env, a)
            total += r
            if done:
                break
        totals.append(total)
    return float(np.mean(totals)), float(np.std(totals))

# Hyperparameter grid (10 runs)
configs = [
    {"learning_rate":1e-3, "gamma":0.99, "hidden":128, "episodes":800},
    {"learning_rate":5e-4, "gamma":0.98, "hidden":64, "episodes":1200},
    {"learning_rate":2e-3, "gamma":0.95, "hidden":128, "episodes":600},
    {"learning_rate":1e-3, "gamma":0.97, "hidden":256, "episodes":800},
    {"learning_rate":5e-4, "gamma":0.99, "hidden":128, "episodes":1000},
    {"learning_rate":2e-4, "gamma":0.99, "hidden":64, "episodes":1200},
    {"learning_rate":1e-3, "gamma":0.9, "hidden":128, "episodes":800},
    {"learning_rate":8e-4, "gamma":0.98, "hidden":256, "episodes":800},
    {"learning_rate":1e-3, "gamma":0.995, "hidden":128, "episodes":500},
    {"learning_rate":5e-4, "gamma":0.96, "hidden":64, "episodes":1000},
]

for idx, cfg in enumerate(configs, start=1):
    run_name = f"reinforce_run{idx}"
    print(f"\n=== {run_name} - {cfg} ===")

    env = StrokeDetectionEnv()
    obs_size = env.observation_space.shape[0]
    n_actions = env.action_space.n

    policy = PolicyNet(obs_size=obs_size, n_actions=n_actions, hidden=cfg["hidden"]).to(device)
    optimizer = optim.Adam(policy.parameters(), lr=cfg["learning_rate"])

    episodes = cfg["episodes"]
    gamma = cfg["gamma"]
    timesteps_done = 0
    all_episode_rewards = []

    # training loop
    for ep in range(1, episodes+1):
        obs = safe_reset(env)
        log_probs = []
        rewards = []
        done = False
        while not done:
            obs_t = torch.tensor(obs, dtype=torch.float32, device=device)
            logits = policy(obs_t)
            probs = torch.softmax(logits, dim=-1)
            m = torch.distributions.Categorical(probs)
            a = m.sample()
            log_probs.append(m.log_prob(a))
            obs, r, done, _ = safe_step(env, int(a.item()))
            rewards.append(r)
            timesteps_done += 1
        all_episode_rewards.append(sum(rewards))

        returns = compute_returns(rewards, gamma)
        loss = torch.stack([-lp * R for lp, R in zip(log_probs, returns)]).sum()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if ep % 50 == 0:
            recent_mean = float(np.mean(all_episode_rewards[-50:]))
            print(f"ep {ep}/{episodes} | recent_mean_reward={recent_mean:.2f}")

    # evaluation
    mean_r, std_r = evaluate_policy(policy, StrokeDetectionEnv(), episodes=5)

    # save model
    save_dir = f"models/reinforce/run_{idx}"
    os.makedirs(save_dir, exist_ok=True)
    torch.save(policy.state_dict(), f"{save_dir}/policy_state.pt")

    # log to CSV
    with open(CSV_PATH, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            idx,
            cfg["learning_rate"],
            cfg["gamma"],
            cfg["hidden"],
            cfg["episodes"],
            timesteps_done,
            mean_r
        ])

    print(f"Finished {run_name}: mean_reward={mean_r:.2f} std={std_r:.2f}")
    env = None

print("All REINFORCE runs completed. Logs saved to logs/reinforce_hyperparams.csv")
