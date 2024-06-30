import gymnasium as gym

from stable_baselines3 import PPO
from stable_baselines3.common.atari_wrappers import AtariWrapper
from stable_baselines3.common.evaluation import evaluate_policy

# env = gym.make("QbertNoFrameskip-v4")
env = gym.make("QbertNoFrameskip-v4", render_mode="human")
env = AtariWrapper(env)

# model = PPO("CnnPolicy",
#             env,
#             n_steps=128,
#             batch_size=256,
#             n_epochs=4,
#             learning_rate=2.5e-4,
#             clip_range=0.1,
#             ent_coef=0.01,
#             verbose=1,
#             tensorboard_log="./ppo_qbert_tensorboard_log/")
# trained_model = model.learn(total_timesteps=60_000_000, log_interval=4)
# trained_model.save("rl/history/baselines/ppo_qbert_60m_atari_wrapper")

model = PPO.load("rl/history/baselines/ppo_qbert_60m_atari_wrapper")
mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10)
print(f"mean_reward:{mean_reward:.2f} +/- {std_reward:.2f}")
