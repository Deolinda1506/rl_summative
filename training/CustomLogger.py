from stable_baselines3.common.callbacks import BaseCallback
import numpy as np

class StrokeLoggerCallback(BaseCallback):
    def __init__(self, verbose=0):
        super(StrokeLoggerCallback, self).__init__(verbose)
        self.episode_rewards = []
        self.episode_lengths = []

    def _on_step(self) -> bool:
        infos = self.locals.get('infos', [])
        for info in infos:
            if 'episode' in info:
                episode_reward = info['episode']['r']
                episode_length = info['episode']['l']

                self.episode_rewards.append(episode_reward)
                self.episode_lengths.append(episode_length)

                if self.verbose:
                    print(f"Episode {len(self.episode_rewards)} ended:")
                    print(f"  Reward: {episode_reward}")
                    print(f"  Length: {episode_length}")

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
