# training/plot_hyperparams.py

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Create a folder to save figures
os.makedirs("figures", exist_ok=True)

# Function to generate graphs from CSV
def plot_hyperparam_results(csv_file, algorithm_name):
    if not os.path.exists(csv_file):
        print(f"⚠ CSV file not found: {csv_file}")
        return

    # Load CSV
    df = pd.read_csv(csv_file)

    # Normalize column names: lowercase, strip spaces
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
    plt.figure(figsize=(12,6))
    sns.barplot(x="run_name", y=reward_col, data=df, palette="Blues_d")
    plt.xticks(rotation=45)
    plt.title(f"{algorithm_name} Hyperparameter Run Comparison ({reward_col})")
    plt.ylabel("Reward")
    plt.xlabel("Run Name")
    plt.tight_layout()
    plt.savefig(f"figures/{algorithm_name}_reward_bar.png", dpi=300)
    plt.close()

    # Scatter plot: Learning Rate vs Reward (if columns exist)
    plt.figure(figsize=(10,6))
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

    print(f" Plots generated for {algorithm_name} and saved in 'figures/'")

# Run for all algorithms
logs = {
    "DQN": "logs/dqn_hyperparams.csv",
    "PPO": "logs/ppo_hyperparams.csv",
    "A2C": "logs/a2c_hyperparams.csv",
    "REINFORCE": "logs/reinforce_hyperparams.csv"
}

for algo, csv_path in logs.items():
    plot_hyperparam_results(csv_path, algo)
