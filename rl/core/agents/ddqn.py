import timeit
import numpy as np
from gymnasium import Env
from keras.models import Sequential

from core.agents.base import Agent
from core.brain import Brain
from core.policy import Policy
from core.memory import Memory


class DDQNAgent(Agent):
    def __init__(self, env: Env, model: Sequential, policy: Policy,
                 memory: Memory, create_env, preprocessing=None,
                 reward_fn=None, episodes=2000, episode_start=0, t_warm_up=0,
                 batch_size=32, gamma=0.99, alpha=0.6, beta=0.4,
                 target_update_steps=5000, replay_steps=4,
                 prioritized_exp_replay=False) -> None:
        super().__init__(env.spec.id, create_env, model, policy, preprocessing,
                         reward_fn)
        self.env = env
        self.brain = Brain(model)
        self.target_brain = self.brain.copy()
        self.policy = policy
        self.memory = memory

        self.reward_fn = reward_fn

        self.episodes = episodes
        self.episode_start = episode_start
        self.current_episode = episode_start
        self.t_warm_up = t_warm_up
        self.batch_size = batch_size
        self.gamma = gamma
        self.alpha = alpha
        self.beta = beta
        self.beta_step = (1 - beta) / (episodes - episode_start)
        self.target_update_steps = target_update_steps
        self.replay_steps = replay_steps

        self.prioritized_exp_replay = prioritized_exp_replay

    def fit(self) -> None:
        try:
            state, info = self.env.reset()
            for i in range(self.t_warm_up):
                print(f'Warm-up step {i+1}/{self.t_warm_up}.')
                action = self.action(state)
                step = self.env.step(action)
                next_state, reward, terminated, truncated, next_info = step

                if self.reward_fn is not None:
                    reward = self.reward_fn(state, info, next_state,
                                            reward, terminated, truncated,
                                            next_info)

                done = terminated or truncated

                self.remember(state, action, reward, next_state, done)

                state = next_state
                info = next_info

                if done:
                    state, info = self.env.reset()

            metrics = []
            steps = 0
            for i in range(self.episode_start, self.episodes):
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

                    if steps % self.target_update_steps == 0:
                        print('Updating target function.')
                        self.target_brain = self.brain.copy()

                self.beta += self.beta_step
                self.current_episode = i

                if i % 25 == 0:
                    self.metrics = metrics
                    self.save('rl/history', overwrite=True)

                score = self.score()
                print(f'Episode score: {score}.')

                self.policy.update(i)
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
            print(f'An error occurred during training: {e}.')
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
        q_next = self.brain.forward(next_state)
        efr = self.brain.forward(next_state)[np.argmax(q_next)]
        return reward + self.gamma * efr - q

    def experience_replay(self) -> dict[str, float]:
        if len(self.memory) < self.batch_size:
            return

        states = []
        targets = []

        batch = self.memory.sample(self.batch_size)
        time = timeit.default_timer()
        for _, state, action, reward, next_state, done in batch:
            target = self.brain.forward(state)
            target[action] = reward

            if not done:
                efr = np.amax(self.target_brain.forward(next_state))
                target[action] += self.gamma * efr

            states.append(state)
            targets.append(target)

        print(f'Gathered batch in {timeit.default_timer() - time} seconds.')

        time = timeit.default_timer()
        metrics = self.brain.backward(np.array(states), np.array(targets))
        print(f'Backward pass in {timeit.default_timer() - time} seconds.')

        return metrics

    def prioritized_experience_replay(self) -> dict[str, float]:
        if len(self.memory) < self.batch_size:
            return

        batch = self.memory.prioritized_sample(self.batch_size, self.alpha,
                                               self.beta)

        time = timeit.default_timer()

        states = [memory[1] for memory in batch]
        targets = self.brain.forward_batch(states)

        next_states = [memory[4] for memory in batch]
        efrs = self.target_brain.forward_batch(next_states)

        for i, (_, state, action, reward, _, done, _) in enumerate(batch):
            target = targets[i]
            target[action] = reward

            if not done:
                efr = np.amax(efrs[i])
                target[action] += self.gamma * efr

            targets[i] = target

        weights = [memory[6] for memory in batch]

        print(f'Gathered batch in {timeit.default_timer() - time} seconds.')

        time = timeit.default_timer()

        metrics = self.brain.backward(np.array(states), np.array(targets),
                                      weights=np.array(weights))

        print(f'Backward pass in {timeit.default_timer() - time} seconds.')

        for i, state, action, reward, next_state, done, _ in batch:
            error = self.get_error(state, action, reward, next_state)
            self.memory.set_error(i, error)

        return metrics

    def get_config(self) -> dict:
        return {
            'env': self.env.spec.id,
            'algorithm': 'DDQN',
            'model': self.brain.config(),
            'memory': self.memory.config(),
            'policy': self.policy.config(),
            'episodes': self.episodes,
            'last_episode': self.current_episode,
            'batch_size': self.batch_size,
            'gamma': self.gamma,
            'alpha': self.alpha,
            'beta': self.beta,
            'target_update_steps': self.target_update_steps,
            'replay_steps': self.replay_steps,
            'prioritized_exp_replay': self.prioritized_exp_replay
        }
