import gymnasium as gym
from gymnasium import Env
from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv2D
from keras.optimizers import Adam

from core.agents.dqn import DQNAgent
from core.memory import SequentialMemory
from core.policy import BoltzmannPolicy
from core.wrappers import (
    ClipRewardEnv, EpisodicLifeEnv, MaxAndSkipEnv, WarpFrame
)


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


def create_env(visualize=False) -> Env:
    id = "ALE/Qbert-v5"
    if visualize:
        env = gym.make(id, render_mode='human')
    else:
        env = gym.make(id)

    env = MaxAndSkipEnv(env, skip=4)
    env = EpisodicLifeEnv(env)
    env = WarpFrame(env, width=84, height=84)
    env = ClipRewardEnv(env)

    return env


if __name__ == "__main__":
    env = create_env()
    state_shape = env.observation_space.shape
    n_actions = env.action_space.n

    model = build_model(state_shape, n_actions)

    episodes = 2000
    memory = SequentialMemory(50000)
    policy = BoltzmannPolicy(1.0, 0.1, 0.9985)
    agent = DQNAgent(env, model, policy, memory, create_env,
                     episodes=episodes, prioritized_exp_replay=True)

    agent.fit()
    agent.save('rl/history', overwrite=True)
