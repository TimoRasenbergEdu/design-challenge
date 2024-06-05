import numpy as np
from gymnasium import Env
from keras.models import Sequential

from core.agents.base import Agent
from core.brain import Brain
from core.policy import Policy
from core.memory import Memory


class DQNAgent(Agent):
    def __init__(self, env: Env, model: Sequential, policy: Policy,
                 memory: Memory, preprocessing=None, reward_fn=None,
                 episodes=2000, episode_start=0, batch_size=64, gamma=0.99,
                 alpha=0.6, beta=0.4, replay_steps=4,
                 prioritized_exp_replay=False) -> None:
        super().__init__(env.spec.id, model, policy, preprocessing, reward_fn)
        self.env = env
        self.brain = Brain(model)
        self.policy = policy
        self.memory = memory

        self.reward_fn = reward_fn

        self.episodes = episodes
        self.episode_start = episode_start
        self.current_episode = episode_start
        self.batch_size = batch_size
        self.gamma = gamma
        self.alpha = alpha
        self.beta = beta
        self.beta_step = (1 - beta) / (episodes - episode_start)
        self.replay_steps = replay_steps

        self.prioritized_exp_replay = prioritized_exp_replay

    def fit(self) -> None:
        try:
            metrics = []
            steps = 0
            for i in range(self.episodes):
                print(f'Episode {i+1}/{self.episodes}.')

                state, info = self.env.reset()

                episode_metrics = []
                done = False
                while not done:
                    action = self.action(state)
                    step = self.env.step(action)
                    next_state, reward, terminated, truncated, next_info = step

                    if self.reward_fn is not None:
                        reward = self.reward_fn(state, info, next_state,
                                                reward, terminated, truncated,
                                                next_info)

                    done = terminated or truncated

                    self.remember(state, action, reward, next_state, done)

                    if steps % self.replay_steps == 0:
                        if self.prioritized_exp_replay:
                            step_metrics = self.prioritized_experience_replay()
                        else:
                            step_metrics = self.experience_replay()

                        episode_metrics.append(step_metrics)

                    state = next_state
                    info = next_info
                    steps += 1

                self.beta += self.beta_step
                self.current_episode = i

                if i % 25 == 0:
                    self.metrics = metrics
                    self.save('rl/history', overwrite=True)

                score = self.score()
                print(f'Episode score: {score}.')

                self.policy.update()
                print(f'Policy updated: {self.policy}.')

                metrics.append(
                    {
                        'metrics': episode_metrics,
                        'score': score
                    }
                )
        except KeyboardInterrupt:
            print('Training interrupted.')
        except Exception as e:
            raise e
        finally:
            self.metrics = metrics

    def remember(self, state, action, reward, next_state, done) -> None:
        self.memory.append_memory((state, action, reward, next_state, done))

        if self.prioritized_exp_replay:
            error = self.get_error(state, action, reward, next_state)
            self.memory.append_error(error)

    def get_error(self, state, action, reward, next_state) -> None:
        q = self.brain.forward(state)[action]
        efr = np.amax(self.brain.forward(next_state))
        return reward + self.gamma * efr - q

    def experience_replay(self) -> dict[str, float]:
        if len(self.memory) < self.batch_size:
            return

        states = []
        targets = []

        batch = self.memory.sample(self.batch_size)
        for _, state, action, reward, next_state, done in batch:
            target = self.brain.forward(state)
            target[action] = reward

            if not done:
                efr = np.amax(self.brain.forward(next_state))
                target[action] += self.gamma * efr

            states.append(state)
            targets.append(target)

        return self.brain.backward(np.array(states), np.array(targets))

    def prioritized_experience_replay(self) -> dict[str, float]:
        if len(self.memory) < self.batch_size:
            return

        states = []
        targets = []
        weights = []

        batch = self.memory.prioritized_sample(self.batch_size, self.alpha,
                                               self.beta)
        for _, state, action, reward, next_state, done, weight in batch:
            target = self.brain.forward(state)
            target[action] = reward

            if not done:
                efr = np.amax(self.brain.forward(next_state))
                target[action] += self.gamma * efr

            states.append(state)
            targets.append(target)
            weights.append(weight)

        metrics = self.brain.backward(np.array(states), np.array(targets),
                                      weights=np.array(weights))

        for i, state, action, reward, next_state, done, _ in batch:
            error = self.get_error(state, action, reward, next_state)
            self.memory.set_error(i, error)

        return metrics

    def get_config(self) -> dict:
        return {
            'env': self.env.spec.id,
            'algorithm': 'DQN',
            'model': self.brain.config(),
            'memory': self.memory.config(),
            'policy': self.policy.config(),
            'episodes': self.episodes,
            'last_episode': self.current_episode,
            'batch_size': self.batch_size,
            'gamma': self.gamma,
            'alpha': self.alpha,
            'beta': self.beta,
            'replay_steps': self.replay_steps,
            'prioritized_exp_replay': self.prioritized_exp_replay
        }
