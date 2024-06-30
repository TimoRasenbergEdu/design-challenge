import os
import json
from datetime import datetime
from keras import Sequential

from core.policy import Policy, GreedyPolicy
from core.brain import Brain


class Agent:
    def __init__(self, create_env, model: Sequential, policy: Policy) -> None:
        self.create_env = create_env
        self.brain = Brain(model)
        self.policy = policy

    def action(self, state) -> int:
        return self.policy.action(self.brain.forward(state))

    def score(self, visualize=False) -> float:
        policy = GreedyPolicy()

        if visualize:
            env = self.create_env(render=True)
        else:
            env = self.create_env()

        state, _ = env.reset()

        cum_reward = 0
        while True:
            action = policy.action(self.brain.forward(state))
            next_state, reward, terminated, truncated, _ = env.step(action)

            cum_reward += reward

            state = next_state

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
