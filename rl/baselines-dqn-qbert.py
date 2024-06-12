import gymnasium as gym

from stable_baselines3 import DQN

env = gym.make("ALE/Qbert-v5", render_mode="human")

model = DQN("CnnPolicy", env, verbose=1)
model.learn(total_timesteps=1000000, log_interval=4)

obs, info = env.reset()
while True:
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        obs, info = env.reset()
