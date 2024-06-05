import gymnasium as gym
from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv2D
from keras.optimizers import Adam

from core.agents.dqn import DQNAgent
from core.memory import SequentialMemory
from core.policy import BoltzmannPolicy


def build_model(state_shape, n_actions):
    model = Sequential([
        Conv2D(32, (8, 8), strides=(4, 4), activation='relu',
               input_shape=(*state_shape, 1)),
        Conv2D(64, (4, 4), strides=(2, 2), activation='relu'),
        Conv2D(64, (3, 3), strides=(1, 1), activation='relu'),
        Flatten(),
        Dense(1024, activation='relu'),
        Dense(1024, activation='relu'),
        Dense(n_actions)
    ])

    model.compile(optimizer=Adam(1e-3), loss='mse')

    return model


def preprocess_input(input):
    return input / 255.0


def reward_fn(state, info, next_state, reward, terminated, truncated,
              next_info) -> float:
    if next_info['lives'] < info['lives']:
        return reward - 100


if __name__ == "__main__":
    env = gym.make("ALE/Qbert-v5", obs_type='grayscale')
    state_shape = env.observation_space.shape
    n_actions = env.action_space.n

    model = build_model(state_shape, n_actions)

    episodes = 2000
    memory = SequentialMemory(50000)
    policy = BoltzmannPolicy(1.0, 0.9985)
    agent = DQNAgent(env, model, policy, memory, episodes=episodes,
                     preprocessing=preprocess_input,
                     reward_fn=reward_fn,
                     prioritized_exp_replay=True)

    agent.fit()
    agent.save('rl/history', overwrite=True)
