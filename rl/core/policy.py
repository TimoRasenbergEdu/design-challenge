import numpy as np


class Policy:
    def __init__(self) -> None:
        pass

    def action(self, action: np.ndarray) -> int:
        raise NotImplementedError

    def update(self, step: int) -> None:
        raise NotImplementedError

    def config(self) -> dict:
        raise NotImplementedError

    def __str__(self) -> str:
        return str(self.config())


class GreedyPolicy(Policy):
    def action(self, q_values: np.ndarray) -> int:
        return np.argmax(q_values)

    def config(self) -> dict:
        return {
            'type': 'Greedy'
        }


class EpsilonGreedyPolicy(Policy):
    def __init__(self, epsilon: float, n_actions: int) -> None:
        self.epsilon = epsilon
        self.n_actions = n_actions

    def action(self, q_values: np.ndarray) -> int:
        if np.random.random() < self.epsilon:
            return np.random.randint(self.n_actions)
        else:
            return np.argmax(q_values)

    def config(self) -> dict:
        return {
            'type': 'EpsilonGreedy',
            'epsilon': self.epsilon
        }


class EpsilonDecayPolicy(Policy):
    def __init__(self, epsilon_init: float, epsilon_min: float,
                 decay_rate: float, n_actions: int) -> None:
        self.epsilon_init = epsilon_init
        self.epsilon_current = epsilon_init
        self.epsilon_min = epsilon_min
        self.decay_rate = decay_rate
        self.n_actions = n_actions

    def action(self, q_values: np.ndarray) -> int:
        epsilon_greedy = EpsilonGreedyPolicy(self.epsilon_current,
                                             self.n_actions)
        return epsilon_greedy.action(q_values)

    def update(self, step: int) -> None:
        self.epsilon_current = max(
            self.epsilon_current * (self.decay_rate ** step),
            self.epsilon_min
        )

    def config(self) -> dict:
        return {
            'type': 'EpsilonDecay',
            'epsilon_init': self.epsilon_init,
            'epsilon_current': self.epsilon_current,
            'epsilon_min': self.epsilon_min,
            'decay_rate': self.decay_rate
        }


class SoftmaxPolicy(Policy):
    def action(self, q_values: np.ndarray) -> int:
        q_values = q_values - np.max(q_values)

        probabilities = np.exp(q_values)
        probabilities /= np.sum(probabilities)

        return np.random.choice(len(q_values), p=probabilities)

    def config(self) -> dict:
        return {
            'type': 'Softmax'
        }


class BoltzmannPolicy(Policy):
    def __init__(self, tau_init: float, tau_min: float, decay_rate: float):
        self.tau_init = tau_init
        self.tau_current = tau_init
        self.tau_min = tau_min
        self.decay_rate = decay_rate

    def action(self, q_values: np.ndarray) -> int:
        q_values = q_values - np.max(q_values)

        probabilities = np.exp(q_values / self.tau_current)
        probabilities /= np.sum(probabilities)

        return np.random.choice(len(q_values), p=probabilities)

    def update(self, step: int) -> None:
        self.tau_current = max(
            self.tau_init * (self.decay_rate ** step),
            self.tau_min
        )

    def config(self) -> dict:
        return {
            'type': 'Boltzmann',
            'tau_init': self.tau_init,
            'tau_current': self.tau_current,
            'tau_min': self.tau_min,
            'decay_rate': self.decay_rate
        }
