import numpy as np
import gymnasium as gym
from gymnasium.wrappers import ResizeObservation, FrameStack
from PIL import Image
from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv2D
from keras.optimizers import Adam


class TransposeFrame(gym.ObservationWrapper):
    def __init__(self, env):
        super(TransposeFrame, self).__init__(env)
        self.observation_space = gym.spaces.Box(
            low=0,
            high=255,
            shape=(126, 96, 12),
            dtype=np.uint8
        )

    def observation(self, observation):
        return np.transpose(observation, (1, 2, 3, 0)).reshape(126, 96, -1)

# class TransposeFrame(gym.ObservationWrapper):
#     def __init__(self, env):
#         super(TransposeFrame, self).__init__(env)
#         self.observation_space = gym.spaces.Box(
#             low=0,
#             high=255,
#             shape=(210, 160, 12),
#             dtype=np.uint8
#         )

#     def observation(self, observation):
#         return np.transpose(observation, (1, 2, 3, 0)).reshape(210, 160, -1)


def build_model(state_shape, n_actions):
    model = Sequential([
        Conv2D(16, (8, 8), strides=(4, 4), activation='relu',
               input_shape=state_shape),
        Conv2D(32, (4, 4), strides=(2, 2), activation='relu'),
        Flatten(),
        Dense(1024, activation='relu'),
        Dense(1024, activation='relu'),
        Dense(n_actions)
    ])

    model.compile(optimizer=Adam(1e-4), loss='mse')

    return model


env = gym.make("ALE/Qbert-v5", render_mode='rgb_array')
# env = ResizeObservation(env, (126, 96))
# env = FrameStack(env, 4)
# env = TransposeFrame(env)

# print(env.observation_space.shape)

model = build_model(env.observation_space.shape, env.action_space.n)

state, info = env.reset()

lives = info['lives']
for i in range(1):
    j = 0
    while True:
        action = env.action_space.sample()
        state, reward, terminated, truncated, info = env.step(action)

        Image.fromarray(state).save(f'./qbert_{i}_{j}.png')

        if info['lives'] < lives:
            lives = info['lives']
            reward -= 100

        print(state.shape, reward, terminated, truncated, info)

        j += 1

        if terminated or truncated:
            break
