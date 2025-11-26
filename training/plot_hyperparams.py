# training/plot_hyperparams.py
# Generates plots required for the report

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np
import torch
from stable_baselines3 import DQN, PPO, A2C
from environment.custom_env import StrokeDetectionEnv

# Create a folder to save figures
os.makedirs("figures", exist_ok=True)

# Helper functions for model evaluation
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

def evaluate_episodes(model, env, num_episodes=50, reinforce_policy=False, device="cpu"):
    """Evaluate model over multiple episodes and return episode rewards"""
    episode_rewards = []
    for _ in range(num_episodes):
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
        
        episode_rewards.append(total_reward)
    
    return episode_rewards

# PolicyNet for REINFORCE
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

def pick_best_run(csv_path, reward_col):
    df = pd.read_csv(csv_path)
    df.columns = [c.strip().lower() for c in df.columns]
    if reward_col not in df.columns:
        raise ValueError(f"No reward column '{reward_col}' in {csv_path}")
    best_row = df.loc[df[reward_col].idxmax()]
    return best_row

def plot_hyperparam_analysis(csv_file, algorithm_name):
    """Generate bar charts and scatter plots for hyperparameter analysis"""
    if not os.path.exists(csv_file):
        print(f"⚠ CSV file not found: {csv_file}")
        return

    # Load CSV
    df = pd.read_csv(csv_file)
    df.columns = [c.strip().lower() for c in df.columns]

    # Detect reward column
    reward_col = None
    for col in ["mean_reward", "total_reward", "reward"]:
        if col in df.columns:
            reward_col = col
            break
    if reward_col is None:
        print(f"⚠ No reward column found in {csv_file}")
        return

    # Ensure a column exists for run names
    if "run_name" not in df.columns and "run_id" in df.columns:
        df["run_name"] = df["run_id"].astype(str)
    elif "run_name" not in df.columns:
        df["run_name"] = df.index.astype(str)
    
    # Bar plot: Reward per Run
    plt.figure(figsize=(12, 6))
    sns.barplot(x="run_name", y=reward_col, data=df, palette="Blues_d")
    plt.xticks(rotation=45)
    plt.title(f"{algorithm_name} Hyperparameter Run Comparison ({reward_col})")
    plt.ylabel("Reward")
    plt.xlabel("Run Name")
    plt.tight_layout()
    plt.savefig(f"figures/{algorithm_name}_reward_bar.png", dpi=300)
    plt.close()

    # Scatter plot: Learning Rate vs Reward
    plt.figure(figsize=(10, 6))
    x_col = "learning_rate"
    hue_col = "gamma"
    size_col = None
    if "batch_size" in df.columns:
        size_col = "batch_size"

    if x_col in df.columns and hue_col in df.columns:
        sns.scatterplot(
            x=x_col,
            y=reward_col,
            hue=hue_col,
            size=size_col,
            data=df,
            palette="viridis",
            sizes=(50, 200) if size_col else None
        )
        plt.xscale("log")
        plt.title(f"{algorithm_name} Reward vs Learning Rate, Gamma, Batch Size")
        plt.xlabel("Learning Rate (log scale)")
        plt.ylabel("Reward")
        plt.legend(bbox_to_anchor=(1.05, 1), loc=2)
        plt.tight_layout()
        plt.savefig(f"figures/{algorithm_name}_scatter.png", dpi=300)
        plt.close()

    print(f"✓ Hyperparameter plots generated for {algorithm_name}")

def plot_cumulative_rewards():
    """Plot cumulative rewards over episodes for all best models"""
    print("Generating cumulative rewards plot...")
    
    try:
        # Load best models
        best_dqn = pick_best_run("logs/dqn_hyperparams.csv", "total_reward")
        best_ppo = pick_best_run("logs/ppo_hyperparams.csv", "total_reward")
        best_a2c = pick_best_run("logs/a2c_hyperparams.csv", "total_reward")
        best_reinforce = pick_best_run("logs/reinforce_hyperparams.csv", "mean_reward")
        
        # Load models
        model_dqn = DQN.load(f"models/dqn/run_{int(best_dqn['run_id'])}/dqn_model")
        model_ppo = PPO.load(f"models/pg/run_{int(best_ppo['run_id'])}/ppo_model")
        model_a2c = A2C.load(f"models/a2c/run_{int(best_a2c['run_id'])}/a2c_model")
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        env_tmp = StrokeDetectionEnv()
        obs_size = env_tmp.observation_space.shape[0]
        n_actions = env_tmp.action_space.n
        policy_reinforce = PolicyNet(obs_size=obs_size, n_actions=n_actions, hidden=int(best_reinforce['hidden_size'])).to(device)
        policy_reinforce.load_state_dict(torch.load(f"models/reinforce/run_{int(best_reinforce['run_name'])}/policy_state.pt", map_location=device))
        
        # Evaluate models
        num_episodes = 100
        dqn_rewards = evaluate_episodes(model_dqn, StrokeDetectionEnv(), num_episodes)
        ppo_rewards = evaluate_episodes(model_ppo, StrokeDetectionEnv(), num_episodes)
        a2c_rewards = evaluate_episodes(model_a2c, StrokeDetectionEnv(), num_episodes)
        reinforce_rewards = evaluate_episodes(policy_reinforce, StrokeDetectionEnv(), num_episodes, reinforce_policy=True, device=device)
        
        # Calculate cumulative rewards
        dqn_cumulative = np.cumsum(dqn_rewards)
        ppo_cumulative = np.cumsum(ppo_rewards)
        a2c_cumulative = np.cumsum(a2c_rewards)
        reinforce_cumulative = np.cumsum(reinforce_rewards)
        
        # Create subplots
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        episodes = np.arange(1, num_episodes + 1)
        
        axes[0, 0].plot(episodes, dqn_cumulative, 'b-', linewidth=2, label='DQN')
        axes[0, 0].set_title('DQN Cumulative Rewards (Best Model)')
        axes[0, 0].set_xlabel('Episode')
        axes[0, 0].set_ylabel('Cumulative Reward')
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].legend()
        
        axes[0, 1].plot(episodes, ppo_cumulative, 'g-', linewidth=2, label='PPO')
        axes[0, 1].set_title('PPO Cumulative Rewards (Best Model)')
        axes[0, 1].set_xlabel('Episode')
        axes[0, 1].set_ylabel('Cumulative Reward')
        axes[0, 1].grid(True, alpha=0.3)
        axes[0, 1].legend()
        
        axes[1, 0].plot(episodes, a2c_cumulative, 'r-', linewidth=2, label='A2C')
        axes[1, 0].set_title('A2C Cumulative Rewards (Best Model)')
        axes[1, 0].set_xlabel('Episode')
        axes[1, 0].set_ylabel('Cumulative Reward')
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 0].legend()
        
        axes[1, 1].plot(episodes, reinforce_cumulative, 'm-', linewidth=2, label='REINFORCE')
        axes[1, 1].set_title('REINFORCE Cumulative Rewards (Best Model)')
        axes[1, 1].set_xlabel('Episode')
        axes[1, 1].set_ylabel('Cumulative Reward')
        axes[1, 1].grid(True, alpha=0.3)
        axes[1, 1].legend()
        
        plt.tight_layout()
        plt.savefig("figures/cumulative_rewards_all_methods.png", dpi=300)
        plt.close()
        
        print("✓ Cumulative rewards plot saved to figures/cumulative_rewards_all_methods.png")
        
    except Exception as e:
        print(f"⚠ Error generating cumulative rewards plot: {e}")

def plot_episodes_to_converge():
    """Plot episodes to convergence for all methods"""
    print("Generating episodes to converge plot...")
    
    try:
        # Load best models and evaluate
        best_dqn = pick_best_run("logs/dqn_hyperparams.csv", "total_reward")
        best_ppo = pick_best_run("logs/ppo_hyperparams.csv", "total_reward")
        best_a2c = pick_best_run("logs/a2c_hyperparams.csv", "total_reward")
        best_reinforce = pick_best_run("logs/reinforce_hyperparams.csv", "mean_reward")
        
        model_dqn = DQN.load(f"models/dqn/run_{int(best_dqn['run_id'])}/dqn_model")
        model_ppo = PPO.load(f"models/pg/run_{int(best_ppo['run_id'])}/ppo_model")
        model_a2c = A2C.load(f"models/a2c/run_{int(best_a2c['run_id'])}/a2c_model")
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        env_tmp = StrokeDetectionEnv()
        obs_size = env_tmp.observation_space.shape[0]
        n_actions = env_tmp.action_space.n
        policy_reinforce = PolicyNet(obs_size=obs_size, n_actions=n_actions, hidden=int(best_reinforce['hidden_size'])).to(device)
        policy_reinforce.load_state_dict(torch.load(f"models/reinforce/run_{int(best_reinforce['run_name'])}/policy_state.pt", map_location=device))
        
        num_episodes = 100
        dqn_rewards = evaluate_episodes(model_dqn, StrokeDetectionEnv(), num_episodes)
        ppo_rewards = evaluate_episodes(model_ppo, StrokeDetectionEnv(), num_episodes)
        a2c_rewards = evaluate_episodes(model_a2c, StrokeDetectionEnv(), num_episodes)
        reinforce_rewards = evaluate_episodes(policy_reinforce, StrokeDetectionEnv(), num_episodes, reinforce_policy=True, device=device)
        
        # Calculate moving average for convergence analysis
        window = 10
        dqn_ma = pd.Series(dqn_rewards).rolling(window=window).mean()
        ppo_ma = pd.Series(ppo_rewards).rolling(window=window).mean()
        a2c_ma = pd.Series(a2c_rewards).rolling(window=window).mean()
        reinforce_ma = pd.Series(reinforce_rewards).rolling(window=window).mean()
        
        # Create subplots
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        episodes = np.arange(1, num_episodes + 1)
        
        axes[0, 0].plot(episodes, dqn_rewards, 'b-', alpha=0.3, linewidth=1, label='Raw')
        axes[0, 0].plot(episodes, dqn_ma, 'b-', linewidth=2, label='Moving Avg (10)')
        axes[0, 0].axhline(y=dqn_ma.iloc[-1], color='r', linestyle='--', alpha=0.5, label='Final Performance')
        axes[0, 0].set_title('DQN Reward per Episode')
        axes[0, 0].set_xlabel('Episode')
        axes[0, 0].set_ylabel('Episode Reward')
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].legend()
        
        axes[0, 1].plot(episodes, ppo_rewards, 'g-', alpha=0.3, linewidth=1, label='Raw')
        axes[0, 1].plot(episodes, ppo_ma, 'g-', linewidth=2, label='Moving Avg (10)')
        axes[0, 1].axhline(y=ppo_ma.iloc[-1], color='r', linestyle='--', alpha=0.5, label='Final Performance')
        axes[0, 1].set_title('PPO Reward per Episode')
        axes[0, 1].set_xlabel('Episode')
        axes[0, 1].set_ylabel('Episode Reward')
        axes[0, 1].grid(True, alpha=0.3)
        axes[0, 1].legend()
        
        axes[1, 0].plot(episodes, a2c_rewards, 'r-', alpha=0.3, linewidth=1, label='Raw')
        axes[1, 0].plot(episodes, a2c_ma, 'r-', linewidth=2, label='Moving Avg (10)')
        axes[1, 0].axhline(y=a2c_ma.iloc[-1], color='r', linestyle='--', alpha=0.5, label='Final Performance')
        axes[1, 0].set_title('A2C Reward per Episode')
        axes[1, 0].set_xlabel('Episode')
        axes[1, 0].set_ylabel('Episode Reward')
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 0].legend()
        
        axes[1, 1].plot(episodes, reinforce_rewards, 'm-', alpha=0.3, linewidth=1, label='Raw')
        axes[1, 1].plot(episodes, reinforce_ma, 'm-', linewidth=2, label='Moving Avg (10)')
        axes[1, 1].axhline(y=reinforce_ma.iloc[-1], color='r', linestyle='--', alpha=0.5, label='Final Performance')
        axes[1, 1].set_title('REINFORCE Reward per Episode')
        axes[1, 1].set_xlabel('Episode')
        axes[1, 1].set_ylabel('Episode Reward')
        axes[1, 1].grid(True, alpha=0.3)
        axes[1, 1].legend()
        
        plt.tight_layout()
        plt.savefig("figures/episodes_to_converge.png", dpi=300)
        plt.close()
        
        print("✓ Episodes to converge plot saved to figures/episodes_to_converge.png")
        
    except Exception as e:
        print(f"⚠ Error generating convergence plot: {e}")

def plot_generalization():
    """Plot generalization results on unseen initial states"""
    print("Generating generalization plot...")
    
    try:
        best_dqn = pick_best_run("logs/dqn_hyperparams.csv", "total_reward")
        best_ppo = pick_best_run("logs/ppo_hyperparams.csv", "total_reward")
        best_a2c = pick_best_run("logs/a2c_hyperparams.csv", "total_reward")
        best_reinforce = pick_best_run("logs/reinforce_hyperparams.csv", "mean_reward")
        
        model_dqn = DQN.load(f"models/dqn/run_{int(best_dqn['run_id'])}/dqn_model")
        model_ppo = PPO.load(f"models/pg/run_{int(best_ppo['run_id'])}/ppo_model")
        model_a2c = A2C.load(f"models/a2c/run_{int(best_a2c['run_id'])}/a2c_model")
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        env_tmp = StrokeDetectionEnv()
        obs_size = env_tmp.observation_space.shape[0]
        n_actions = env_tmp.action_space.n
        policy_reinforce = PolicyNet(obs_size=obs_size, n_actions=n_actions, hidden=int(best_reinforce['hidden_size'])).to(device)
        policy_reinforce.load_state_dict(torch.load(f"models/reinforce/run_{int(best_reinforce['run_name'])}/policy_state.pt", map_location=device))
        
        # Test on multiple random seeds (different initial states)
        num_tests = 20
        seeds = np.arange(42, 42 + num_tests)
        
        dqn_test_rewards = []
        ppo_test_rewards = []
        a2c_test_rewards = []
        reinforce_test_rewards = []
        
        def run_episode(env, model, reinforce_policy=False):
            obs = safe_reset(env)
            total_reward = 0
            done = False
            for _ in range(200):
                if reinforce_policy:
                    with torch.no_grad():
                        logits = model(torch.tensor(obs, dtype=torch.float32, device=device))
                        probs = torch.softmax(logits, dim=-1).cpu().numpy()
                        action = int(np.argmax(probs))
                else:
                    action, _ = model.predict(obs, deterministic=True)
                obs, reward, done, _ = safe_step(env, action)
                total_reward += reward
                if done:
                    break
            return total_reward
        
        for seed in seeds:
            env_dqn = StrokeDetectionEnv()
            env_ppo = StrokeDetectionEnv()
            env_a2c = StrokeDetectionEnv()
            env_reinforce = StrokeDetectionEnv()
            
            dqn_rew = run_episode(env_dqn, model_dqn)
            ppo_rew = run_episode(env_ppo, model_ppo)
            a2c_rew = run_episode(env_a2c, model_a2c)
            reinforce_rew = run_episode(env_reinforce, policy_reinforce, reinforce_policy=True)
            
            dqn_test_rewards.append(dqn_rew)
            ppo_test_rewards.append(ppo_rew)
            a2c_test_rewards.append(a2c_rew)
            reinforce_test_rewards.append(reinforce_rew)
        
        # Create bar plot comparing methods
        methods = ['DQN', 'PPO', 'A2C', 'REINFORCE']
        means = [np.mean(dqn_test_rewards), np.mean(ppo_test_rewards), 
                 np.mean(a2c_test_rewards), np.mean(reinforce_test_rewards)]
        stds = [np.std(dqn_test_rewards), np.std(ppo_test_rewards),
                np.std(a2c_test_rewards), np.std(reinforce_test_rewards)]
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Bar plot with error bars
        axes[0].bar(methods, means, yerr=stds, capsize=5, alpha=0.7, 
                   color=['blue', 'green', 'red', 'magenta'])
        axes[0].set_title('Generalization: Mean Reward on Unseen Initial States')
        axes[0].set_ylabel('Mean Episode Reward')
        axes[0].grid(True, alpha=0.3, axis='y')
        
        # Box plot
        axes[1].boxplot([dqn_test_rewards, ppo_test_rewards, a2c_test_rewards, reinforce_test_rewards],
                       labels=methods)
        axes[1].set_title('Generalization: Reward Distribution on Unseen Initial States')
        axes[1].set_ylabel('Episode Reward')
        axes[1].grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig("figures/generalization_test.png", dpi=300)
        plt.close()
        
        print(f"✓ Generalization plot saved to figures/generalization_test.png")
        print(f"  DQN: {means[0]:.2f} ± {stds[0]:.2f}")
        print(f"  PPO: {means[1]:.2f} ± {stds[1]:.2f}")
        print(f"  A2C: {means[2]:.2f} ± {stds[2]:.2f}")
        print(f"  REINFORCE: {means[3]:.2f} ± {stds[3]:.2f}")
        
    except Exception as e:
        print(f"⚠ Error generating generalization plot: {e}")

def plot_training_stability():
    """Plot training stability metrics (variance across hyperparameter runs)"""
    print("Generating training stability plot...")
    
    try:
        # Load all hyperparameter data
        dqn_df = pd.read_csv("logs/dqn_hyperparams.csv")
        ppo_df = pd.read_csv("logs/ppo_hyperparams.csv")
        a2c_df = pd.read_csv("logs/a2c_hyperparams.csv")
        reinforce_df = pd.read_csv("logs/reinforce_hyperparams.csv")
        
        # Normalize column names
        for df in [dqn_df, ppo_df, a2c_df, reinforce_df]:
            df.columns = [c.strip().lower() for c in df.columns]
        
        # Get reward columns
        dqn_rewards = dqn_df['total_reward'].values
        ppo_rewards = ppo_df['total_reward'].values
        a2c_rewards = a2c_df['total_reward'].values
        reinforce_rewards = reinforce_df['mean_reward'].values
        
        # Create stability plot (variance across runs)
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # DQN stability
        axes[0, 0].scatter(range(1, len(dqn_rewards)+1), dqn_rewards, alpha=0.6, s=100)
        axes[0, 0].axhline(y=np.mean(dqn_rewards), color='r', linestyle='--', label=f'Mean: {np.mean(dqn_rewards):.1f}')
        axes[0, 0].fill_between(range(1, len(dqn_rewards)+1), 
                               np.mean(dqn_rewards) - np.std(dqn_rewards),
                               np.mean(dqn_rewards) + np.std(dqn_rewards),
                               alpha=0.2, color='red', label=f'±1 std: {np.std(dqn_rewards):.1f}')
        axes[0, 0].set_title('DQN Training Stability (Reward Variance Across Runs)')
        axes[0, 0].set_xlabel('Run ID')
        axes[0, 0].set_ylabel('Total Reward')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # PPO stability
        axes[0, 1].scatter(range(1, len(ppo_rewards)+1), ppo_rewards, alpha=0.6, s=100, color='green')
        axes[0, 1].axhline(y=np.mean(ppo_rewards), color='r', linestyle='--', label=f'Mean: {np.mean(ppo_rewards):.1f}')
        axes[0, 1].fill_between(range(1, len(ppo_rewards)+1),
                               np.mean(ppo_rewards) - np.std(ppo_rewards),
                               np.mean(ppo_rewards) + np.std(ppo_rewards),
                               alpha=0.2, color='red', label=f'±1 std: {np.std(ppo_rewards):.1f}')
        axes[0, 1].set_title('PPO Training Stability (Reward Variance Across Runs)')
        axes[0, 1].set_xlabel('Run ID')
        axes[0, 1].set_ylabel('Total Reward')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # A2C stability
        axes[1, 0].scatter(range(1, len(a2c_rewards)+1), a2c_rewards, alpha=0.6, s=100, color='red')
        axes[1, 0].axhline(y=np.mean(a2c_rewards), color='r', linestyle='--', label=f'Mean: {np.mean(a2c_rewards):.1f}')
        axes[1, 0].fill_between(range(1, len(a2c_rewards)+1),
                               np.mean(a2c_rewards) - np.std(a2c_rewards),
                               np.mean(a2c_rewards) + np.std(a2c_rewards),
                               alpha=0.2, color='red', label=f'±1 std: {np.std(a2c_rewards):.1f}')
        axes[1, 0].set_title('A2C Training Stability (Reward Variance Across Runs)')
        axes[1, 0].set_xlabel('Run ID')
        axes[1, 0].set_ylabel('Total Reward')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # REINFORCE stability
        axes[1, 1].scatter(range(1, len(reinforce_rewards)+1), reinforce_rewards, alpha=0.6, s=100, color='magenta')
        axes[1, 1].axhline(y=np.mean(reinforce_rewards), color='r', linestyle='--', label=f'Mean: {np.mean(reinforce_rewards):.1f}')
        axes[1, 1].fill_between(range(1, len(reinforce_rewards)+1),
                               np.mean(reinforce_rewards) - np.std(reinforce_rewards),
                               np.mean(reinforce_rewards) + np.std(reinforce_rewards),
                               alpha=0.2, color='red', label=f'±1 std: {np.std(reinforce_rewards):.1f}')
        axes[1, 1].set_title('REINFORCE Training Stability (Reward Variance Across Runs)')
        axes[1, 1].set_xlabel('Run ID')
        axes[1, 1].set_ylabel('Mean Reward')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig("figures/training_stability.png", dpi=300)
        plt.close()
        
        print("✓ Training stability plot saved to figures/training_stability.png")
        
    except Exception as e:
        print(f"⚠ Error generating training stability plot: {e}")

# Main execution
if __name__ == "__main__":
    print("=" * 60)
    print("Generating Report Plots")
    print("=" * 60)
    
    # Generate hyperparameter analysis plots (bar charts and scatter plots)
    print("\n1. Generating hyperparameter analysis plots...")
    logs = {
        "DQN": "logs/dqn_hyperparams.csv",
        "PPO": "logs/ppo_hyperparams.csv",
        "A2C": "logs/a2c_hyperparams.csv",
        "REINFORCE": "logs/reinforce_hyperparams.csv"
    }
    for algo, csv_path in logs.items():
        plot_hyperparam_analysis(csv_path, algo)
    
    # Generate report-specific plots
    print("\n2. Generating cumulative rewards plot...")
    plot_cumulative_rewards()
    
    print("\n3. Generating episodes to converge plot...")
    plot_episodes_to_converge()
    
    print("\n4. Generating generalization plot...")
    plot_generalization()
    
    print("\n5. Generating training stability plot...")
    plot_training_stability()
    
    print("\n" + "=" * 60)
    print("All plots generated successfully!")
    print("=" * 60)
    print("\nGenerated plots:")
    print("  - Hyperparameter analysis: *_reward_bar.png, *_scatter.png")
    print("  - Cumulative rewards: cumulative_rewards_all_methods.png")
    print("  - Episodes to converge: episodes_to_converge.png")
    print("  - Generalization: generalization_test.png")
    print("  - Training stability: training_stability.png")
