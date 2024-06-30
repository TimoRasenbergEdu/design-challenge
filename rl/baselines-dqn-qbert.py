import gymnasium as gym

from stable_baselines3 import DQN
from stable_baselines3.common.atari_wrappers import AtariWrapper
from stable_baselines3.common.evaluation import evaluate_policy

# env = gym.make("QbertNoFrameskip-v4")
env = gym.make("QbertNoFrameskip-v4", render_mode="human")
env = AtariWrapper(env)

# model = DQN("CnnPolicy", env, verbose=1, buffer_size=1_000_000,
#             exploration_fraction=0.9,
#             tensorboard_log="./dqn_qbert_tensorboard_log/")
# trained_model = model.learn(total_timesteps=40_000_000, log_interval=4)
# trained_model.save("dqn_qbert_40m_atari_wrapper_ef_0_9")

model = DQN.load("rl/baselines-history/dqn_qbert_40m_atari_wrapper_ef_0_9")
mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10)
# episode_rewards = evaluate_policy(model, env, n_eval_episodes=10,
#                                   return_episode_rewards=True)
print(f"mean_reward:{mean_reward:.2f} +/- {std_reward:.2f}")
# print(episode_rewards)
