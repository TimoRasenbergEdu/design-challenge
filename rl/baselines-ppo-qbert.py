import gymnasium as gym

from stable_baselines3 import PPO
from stable_baselines3.common.atari_wrappers import AtariWrapper
from stable_baselines3.common.evaluation import evaluate_policy

# env = gym.make("QbertNoFrameskip-v4")
env = gym.make("QbertNoFrameskip-v4", render_mode="human")
env = AtariWrapper(env)

# model = PPO("CnnPolicy", env, verbose=1)
# trained_model = model.learn(total_timesteps=40_000_000, log_interval=4)
# trained_model.save("ppo_qbert_40m_atari_wrapper")

model = PPO.load("ppo_qbert_40m_atari_wrapper")
mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10)
print(f"mean_reward:{mean_reward:.2f} +/- {std_reward:.2f}")
