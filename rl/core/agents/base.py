import os
import json
from datetime import datetime
from keras import Sequential

from core.policy import Policy, GreedyPolicy
from core.brain import Brain


class Agent:
    def __init__(self, env_id: str, create_env, model: Sequential,
                 policy: Policy, preprocessing=None, reward_fn=None) -> None:
        self.env_id = env_id
        self.create_env = create_env
        self.brain = Brain(model)
        self.policy = policy
        self.preprocessing = preprocessing
        self.reward_fn = reward_fn

    def action(self, state) -> int:
        if self.preprocessing:
            state = self.preprocessing(state)

        return self.policy.action(self.brain.forward(state))

    def score(self, visualize=False) -> float:
        policy = GreedyPolicy()

        if visualize:
            env = self.create_env(self.env_id, (126, 96), 4, render=True)
        else:
            env = self.create_env(self.env_id, (126, 96), 4)

        state, info = env.reset()

        cum_reward = 0
        while True:
            action = policy.action(self.brain.forward(state))
            next_state, reward, terminated, truncated, next_info = env.step(action)

            if self.reward_fn is not None:
                reward = self.reward_fn(state, info, next_state, reward,
                                        terminated, truncated, next_info)

            cum_reward += reward

            state = next_state
            info = next_info

            if terminated or truncated:
                break

        return cum_reward

    def fit(self) -> None:
        raise NotImplementedError

    def save(self, name: str, overwrite=False) -> None:
        time = datetime.now()
        path = os.path.join(name, time.strftime('%d-%m-%Y-%H-%M-%S'))
        if not os.path.exists(path):
            os.makedirs(path)

        model_path = os.path.join(path, 'model.keras')
        self.brain.save(model_path, overwrite=overwrite)

        metrics_path = os.path.join(path, 'metrics.json')
        with open(metrics_path, 'w') as f:
            json.dump(self.metrics, f, indent=4)

        config_path = os.path.join(path, 'config.json')
        with open(config_path, 'w') as f:
            json.dump(self.get_config(), f, indent=4)

    def get_config(self) -> dict:
        raise NotImplementedError
