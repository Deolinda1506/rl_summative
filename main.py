import os
from stable_baselines3 import DQN, PPO
from gymnasium.wrappers import RecordVideo
from environment.custom_env import StrokeDetectionEnv

# Ensure video folders exist
os.makedirs('rl_agent_videos_dqn', exist_ok=True)
os.makedirs('rl_agent_videos_ppo', exist_ok=True)

# Load trained models
model_dqn = DQN.load("models/dqn/dqn_lr0.0001_gamma0.99_bs64_buf20000.zip")
model_ppo = PPO.load("models/pg/ppo_lr0.0003_gamma0.99_n1024_bs32.zip")


# Create environments wrapped for video recording
env_dqn = RecordVideo(
    StrokeDetectionEnv(render_mode='rgb_array'),
    video_folder='rl_agent_videos_dqn',
    episode_trigger=lambda x: True
)

env_ppo = RecordVideo(
    StrokeDetectionEnv(render_mode='rgb_array'),
    video_folder='rl_agent_videos_ppo',
    episode_trigger=lambda x: True
)

# Evaluation function
def evaluate_and_record(model, env, episodes=3, model_name='model'):
    for episode in range(episodes):
        obs, _ = env.reset()
        total_reward = 0
        for step in range(200):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            total_reward += reward
            if done:
                break
        print(f"{model_name} Episode {episode+1}: Total Reward = {total_reward:.2f}")
    env.close()

# Run evaluation and record videos
if __name__ == "__main__":
    print("Recording DQN agent...")
    evaluate_and_record(model_dqn, env_dqn, model_name='DQN')

    print("Recording PPO agent...")
    evaluate_and_record(model_ppo, env_ppo, model_name='PPO')
