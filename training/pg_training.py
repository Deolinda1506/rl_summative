from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from CustomLogger import StrokeLoggerCallback
from environment.custom_env import StrokeDetectionEnv

# Wrap the environment
env = DummyVecEnv([lambda: Monitor(StrokeDetectionEnv())])

# Optional: custom callback for logging
logger_callback = StrokeLoggerCallback(verbose=1)

# Create PPO model
model = PPO(
    "MlpPolicy",
    env,
    verbose=1,
    learning_rate=0.0003,
    gamma=0.95,
    batch_size=32,
    n_steps=2048
)

# Train the model
model.learn(total_timesteps=50000, callback=logger_callback)

# Save the trained model
model.save("models/pg/stroke_ppo_model")

# Close environment
env.close()
