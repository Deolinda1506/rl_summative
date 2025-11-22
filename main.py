# evaluate_and_record_stroke.py
from stable_baselines3 import DQN, PPO, A2C
from gymnasium.wrappers import RecordVideo
from environment.custom_env import StrokeDetectionEnv  # Make sure your environment is renamed

# Load trained models (replace with your actual file paths)
model_dqn = DQN.load("stroke_dqn_model.zip")
model_ppo = PPO.load("stroke_ppo_model.zip")
model_a2c = A2C.load("stroke_a2c_model.zip")  # Optional if you trained A2C

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

env_a2c = RecordVideo(
    StrokeDetectionEnv(render_mode='rgb_array'),
    video_folder='rl_agent_videos_a2c',
    episode_trigger=lambda x: True
)

# Evaluation and recording function
def evaluate_and_record(model, env, episodes=3, model_name='model'):
    for episode in range(episodes):
        obs, _ = env.reset()
        total_reward = 0
        for step in range(200):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, _, _ = env.step(action)
            total_reward += reward
            if done:
                break
        print(f"{model_name} Episode {episode+1}: Total Reward = {total_reward:.2f}")
    env.close()

# Run evaluation and save videos
if __name__ == "__main__":
    print("Recording DQN agent...")
    evaluate_and_record(model_dqn, env_dqn, model_name='DQN')

    print("Recording PPO agent...")
    evaluate_and_record(model_ppo, env_ppo, model_name='PPO')

    print("Recording A2C agent...")
    evaluate_and_record(model_a2c, env_a2c, model_name='A2C')
