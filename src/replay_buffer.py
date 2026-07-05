import numpy as np

from src import config
from src.environment import make_env

class ReplayBuffer:
    def __init__(self):
        self.episodes = []

    def add_episode(self, obs, actions, rewards):
        self.episodes.append((
            np.asarray(obs, dtype=np.uint8),      # (T+1, H, W, 3)
            np.asarray(actions, dtype=np.int64),  # (T,)
            np.asarray(rewards, dtype=np.float32),  # (T,)
        ))

    def __len__(self):
        return len(self.episodes)

    def num_transitions(self):
        return sum(len(a) for _, a, _ in self.episodes)

    def num_successes(self):
        return sum(1 for _, _, r in self.episodes if r.sum() > 0)


def collect_random(buffer, num_episodes, seed0=0):
    rng = np.random.default_rng(config.SEED)
    env = make_env()
    for e in range(num_episodes):
        obs, _ = env.reset(seed=seed0 + e)
        frames, actions, rewards = [obs], [], []
        done = False
        while not done:
            action = int(rng.choice(config.ACTIONS))
            obs, reward, terminated, truncated, _ = env.step(action)
            frames.append(obs)
            actions.append(action)
            rewards.append(reward)
            done = terminated or truncated
        buffer.add_episode(frames, actions, rewards)
    env.close()
    return buffer
