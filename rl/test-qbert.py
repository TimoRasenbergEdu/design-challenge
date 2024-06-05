import gymnasium as gym


env = gym.make("ALE/Qbert-v5", render_mode='human')

state, info = env.reset()

lives = info['lives']
for _ in range(100):
    while True:
        action = env.action_space.sample()
        state, reward, terminated, truncated, info = env.step(action)

        if info['lives'] < lives:
            lives = info['lives']
            reward -= 100

        print(reward, terminated, truncated, info)

        if terminated or truncated:
            break
