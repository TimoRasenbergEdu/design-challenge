import os
import json
import matplotlib.pyplot as plt

src_path = 'rl/history'
for dir in os.listdir(src_path):
    if dir == '.old':
        continue

    if dir != 'DDQN-500':
        continue

    path = os.path.join(src_path, dir)
    with (open(os.path.join(path, 'metrics.json')) as f_metrics,
          open(os.path.join(path, 'config.json')) as f_config):
        metrics = json.load(f_metrics)
        config = json.load(f_config)

        layers = config['model']['layers'][1:-1]
        name = ''
        for i, layer in enumerate(layers):
            if layer['type'] == 'Dense':
                name += f"{layer['units']}"
            elif layer['type'] == 'Dropout':
                name += "D"

            if i != len(layers) - 1:
                name += '-'

        rewards = []
        loss = []
        for episode in metrics:
            rewards.append(episode['cum_reward'])

            epsiode_loss = []
            loss_metrics = episode['metrics']
            for step in loss_metrics:
                if step is not None:
                    epsiode_loss.append(*step['loss'])

            if len(epsiode_loss) > 0:
                loss.append(sum(epsiode_loss) / len(epsiode_loss))
            else:
                loss.append(0)

        plt.plot(loss)
        plt.xlabel('Episode')
        plt.ylabel('Loss')
        plt.show()

        plt.savefig(os.path.join(path, f'{name}-loss.png'))
        plt.clf()

        plt.plot(rewards)
        plt.xlabel('Episode')
        plt.ylabel('Reward')
        plt.show()

        plt.savefig(os.path.join(path, f'{name}-reward.png'))
        plt.clf()
