import numpy as np
import gymnasium as gym
from gymnasium import Env
from gymnasium.wrappers import ResizeObservation, FrameStack
from keras import Input
from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv2D
from keras.optimizers import Adam

from core.agents.ddqn import DDQNAgent
from core.memory import SequentialMemory
from core.policy import EpsilonDecayPolicy
# from core.agents.factory import AgentFactory


class TransposeFrame(gym.ObservationWrapper):
    def __init__(self, env, size, frame_stack) -> None:
        super(TransposeFrame, self).__init__(env)
        self.size = size
        self.frame_stack = frame_stack
        self.observation_space = gym.spaces.Box(
            low=0,
            high=255,
            shape=(size[1], size[0]),
            dtype=np.uint8
        )

    def observation(self, observation):
        return np.transpose(observation, (1, 0))


def build_model(state_shape, n_actions):
    model = Sequential([
        Conv2D(32, (8, 8), strides=(4, 4), activation='relu',
               input_shape=state_shape),
        Conv2D(32, (4, 4), strides=(2, 2), activation='relu'),
        Conv2D(64, (3, 3), strides=(1, 1), activation='relu'),
        Flatten(),
        Dense(512, activation='relu'),
        Dense(n_actions)
    ])

    model.compile(optimizer=Adam(1e-4), loss='mse')

    return model


def create_env(id: str, size: tuple[int, int], frame_stack: int,
               visualize=False) -> Env:
    if visualize:
        env = gym.make(id, render_mode='human')
    else:
        env = gym.make(id)
    # env = ResizeObservation(env, size)
    # env = FrameStack(env, frame_stack)
    # env = TransposeFrame(env, size, frame_stack)

    return env


def preprocess_input(input):
    return input / 255.0


def reward_fn(state, info, next_state, reward, terminated, truncated,
              next_info) -> float:
    if next_info['lives'] < info['lives']:
        return reward - 100

    return reward


if __name__ == "__main__":
    env = create_env("ALE/Qbert-v5", (84, 84), 4)

    state_shape = env.observation_space.shape
    n_actions = env.action_space.n

    model = build_model(state_shape, n_actions)

    episodes = 2000
    memory = SequentialMemory(100000)
    policy = EpsilonDecayPolicy(1.0, 0.05, 0.1, n_actions)
    agent = DDQNAgent(env, model, policy, memory, create_env,
                      episodes=episodes, preprocessing=preprocess_input,
                      reward_fn=reward_fn, prioritized_exp_replay=True)

    # agent = AgentFactory.load('rl/history/10-06-2024-13-37-44',
    #                           create_env=create_env,
    #                           preprocessing=preprocess_input,
    #                           reward_fn=reward_fn)

    agent.fit()
    agent.save('rl/history', overwrite=True)
