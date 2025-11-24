from stable_baselines3.common.callbacks import BaseCallback
import numpy as np
import csv
import os

class StrokeLoggerCallback(BaseCallback):
    def __init__(self, verbose=0, log_csv_path=None, hyperparams=None):
        super().__init__(verbose)
        self.episode_rewards = []
        self.episode_lengths = []
        self.log_csv_path = log_csv_path
        self.hyperparams = hyperparams or {}

    def _on_step(self) -> bool:
        infos = self.locals.get('infos', [])
        for info in infos:
            if 'episode' in info:
                episode_reward = info['episode']['r']
                episode_length = info['episode']['l']

                self.episode_rewards.append(episode_reward)
                self.episode_lengths.append(episode_length)

                if self.verbose:
                    print(f"Episode {len(self.episode_rewards)} ended: Reward={episode_reward}, Length={episode_length}")
        return True

    def _on_training_end(self) -> None:
        total_episodes = len(self.episode_rewards)
        if total_episodes == 0:
            print("No episodes completed. Consider increasing total_timesteps.")
            return

        avg_reward = np.mean(self.episode_rewards)
        avg_length = np.mean(self.episode_lengths)

        print("\n==== Training Summary ====")
        print(f"Total Episodes: {total_episodes}")
        print(f"Average Reward per Episode: {avg_reward:.2f}")
        print(f"Average Episode Length: {avg_length:.2f}")
        print(f"Final Episode Reward: {self.episode_rewards[-1]:.2f}")
        print("==========================\n")

        # Save to CSV if path provided
        if self.log_csv_path:
            file_exists = os.path.isfile(self.log_csv_path)
            with open(self.log_csv_path, mode='a', newline='') as csv_file:
                writer = csv.DictWriter(csv_file, fieldnames=['run_name', 'total_episodes', 'avg_reward', 'avg_length', *self.hyperparams.keys()])
                if not file_exists:
                    writer.writeheader()
                writer.writerow({
                    'run_name': self.model.__class__.__name__,
                    'total_episodes': total_episodes,
                    'avg_reward': avg_reward,
                    'avg_length': avg_length,
                    **self.hyperparams
                })
            print(f"Logged training results to {self.log_csv_path}")
