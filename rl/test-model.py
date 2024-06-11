import os
import keras
import numpy as np
import gymnasium as gym
from gymnasium import Env
from gymnasium.wrappers import ResizeObservation, FrameStack

from core.agents.base import Agent
from core.policy import Policy


class TransposeFrame(gym.ObservationWrapper):
    def __init__(self, env, size, frame_stack) -> None:
        super(TransposeFrame, self).__init__(env)
        self.size = size
        self.frame_stack = frame_stack
        self.observation_space = gym.spaces.Box(
            low=0,
            high=255,
            shape=(size[0], size[1], 3 * frame_stack),
            dtype=np.uint8
        )

    def observation(self, observation):
        return np.transpose(
            observation, (1, 2, 3, 0)
        ).reshape(self.size[0], self.size[1], -1)


def create_env(id: str, size: tuple[int, int], frame_stack: int,
               render=False) -> Env:
    if render:
        env = gym.make(id, render_mode='human')
    else:
        env = gym.make(id)
    env = ResizeObservation(env, size)
    env = FrameStack(env, frame_stack)
    env = TransposeFrame(env, size, frame_stack)

    return env


cum_rewards = []
for dir in ['10-06-2024-02-29-53']:
    path = os.path.join('rl/history', dir)
    model = keras.models.load_model(os.path.join(path, 'model.keras'))

    epsiodes = 1
    model_cum_reward = 0
    for _ in range(epsiodes):
        agent = Agent("ALE/Qbert-v5", create_env, model, Policy())
        cum_reward = agent.score(visualize=True)
        model_cum_reward += cum_reward

    model_cum_reward /= epsiodes
    cum_rewards.append(model_cum_reward)

print(cum_rewards)
