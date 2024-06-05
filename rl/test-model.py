import os
import keras

from core.agents.base import Agent
from core.policy import Policy

cum_rewards = []
for dir in ['05-06-2024-20-12-24']:
    path = os.path.join('rl/history', dir)
    model = keras.models.load_model(os.path.join(path, 'model.keras'))

    epsiodes = 1
    model_cum_reward = 0
    for _ in range(epsiodes):
        agent = Agent("ALE/Qbert-v5", model, Policy())
        cum_reward = agent.score(visualize=True)
        model_cum_reward += cum_reward

    model_cum_reward /= epsiodes
    cum_rewards.append(model_cum_reward)

print(cum_rewards)
