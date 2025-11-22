from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback
from environment.custom_env import StrokeDetectionEnv
import os

# Custom callback for logging (optional)
class StrokeLoggerCallback(BaseCallback):
    def __init__(self, verbose=1):
        super().__init__(verbose)
    def _on_step(self) -> bool:
        return True

# Create directories if not exist
os.makedirs("models/dqn", exist_ok=True)

# Hyperparameter tuning options
learning_rates = [1e-3, 5e-4, 1e-4]
gammas = [0.95, 0.98]
batch_sizes = [32, 64]

for lr in learning_rates:
    for gamma in gammas:
        for batch in batch_sizes:
            env = DummyVecEnv([lambda: Monitor(StrokeDetectionEnv())])
            model_name = f"models/dqn/dqn_lr{lr}_gamma{gamma}_batch{batch}"
            
            model = DQN(
                "MlpPolicy",
                env,
                verbose=1,
                learning_rate=lr,
                gamma=gamma,
                batch_size=batch,
                buffer_size=50000,
                learning_starts=1000,
                exploration_fraction=0.1,
                exploration_final_eps=0.02
            )
            
            model.learn(total_timesteps=50000, callback=StrokeLoggerCallback())
            model.save(model_name)
            env.close()
