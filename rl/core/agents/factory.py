import json
import os
import keras

from core.agents.ddqn import DDQNAgent
from core.agents.dqn import DQNAgent
from core.memory import SequentialMemory
from core.policy import GreedyPolicy, EpsilonGreedyPolicy, EpsilonDecayPolicy, SoftmaxPolicy, BoltzmannPolicy


class AgentFactory:
    @staticmethod
    def load(path: str, create_env, preprocessing=None, reward_fn=None):
        with open(os.path.join(path, 'config.json')) as f:
            config = json.load(f)

        env = create_env(config['env'], (126, 96), 4)
        model = keras.models.load_model(os.path.join(path, 'model.keras'))

        memory = config['memory']
        memory_size = memory['memory_size']
        t_warm_up = memory['last_memory_size']
        memory = SequentialMemory(memory_size)

        episodes = config['episodes']
        episode_start = config['last_episode']
        batch_size = config['batch_size']
        gamma = config['gamma']
        alpha = config['alpha']
        beta = config['beta']
        target_update_steps = config['target_update_steps']
        replay_steps = config['replay_steps']
        prioritized_exp_replay = config['prioritized_exp_replay']

        policy = config['policy']
        policy_type = policy['type']
        if policy_type == 'Greedy':
            policy = GreedyPolicy()
        elif policy_type == 'EpsilonGreedy':
            epsilon = policy['epsilon']
            policy = EpsilonGreedyPolicy(epsilon, env.action_space.n)
        elif policy_type == 'EpsilonDecay':
            epsilon_init = policy['epsilon_init']
            epsilon_min = policy['epsilon_min']
            decay_rate = policy['decay_rate']
            policy = EpsilonDecayPolicy(epsilon_init, epsilon_min, decay_rate,
                                        env.action_space.n)
        elif policy_type == 'Softmax':
            policy = SoftmaxPolicy()
        elif policy_type == 'Boltzmann':
            tau_init = policy['tau_init']
            tau_min = policy['tau_min']
            decay_rate = policy['decay_rate']
            policy = BoltzmannPolicy(tau_init, tau_min, decay_rate)
        else:
            raise ValueError(f'Policy {policy} not supported.')

        algorithm = config['algorithm']
        if algorithm == 'DQN':
            return DQNAgent(env, model, policy, memory, create_env,
                            preprocessing=preprocessing, reward_fn=reward_fn,
                            episodes=episodes, episode_start=episode_start,
                            t_warm_up=t_warm_up, batch_size=batch_size,
                            gamma=gamma, alpha=alpha, beta=beta,
                            replay_steps=replay_steps,
                            prioritized_exp_replay=prioritized_exp_replay)
        elif algorithm == 'DDQN':
            return DDQNAgent(env, model, policy, memory, create_env,
                             preprocessing=preprocessing, reward_fn=reward_fn,
                             episodes=episodes, episode_start=episode_start,
                             t_warm_up=t_warm_up, batch_size=batch_size,
                             gamma=gamma, alpha=alpha, beta=beta,
                             target_update_steps=target_update_steps,
                             replay_steps=replay_steps,
                             prioritized_exp_replay=prioritized_exp_replay)
        else:
            raise ValueError(f'Algorithm {algorithm} not supported.')
