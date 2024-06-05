import numpy as np
from keras import Sequential


class Brain:
    def __init__(self, model: Sequential) -> None:
        self.model = model

    def forward(self, state) -> np.ndarray:
        return self.model(np.array([state]))[0].numpy()

    def backward(self, states, targets, weights=None) -> dict[str, float]:
        return self.model.fit(states, targets, sample_weight=weights).history

    def copy(self) -> 'Brain':
        return Brain(self.model.from_config(self.model.get_config()))

    def save(self, name: str, overwrite=False) -> None:
        self.model.save(name, overwrite=overwrite)

    def config(self) -> dict:
        lr = self.model.optimizer.learning_rate.numpy()
        layers = []
        for layer in self.model.layers:
            name = layer.__class__.__name__
            if name == 'Dense':
                layers.append({
                    'type': name,
                    'units': str(layer.units),
                    'activation': layer.activation.__name__
                })
            elif name == 'Conv2D':
                layers.append({
                    'type': name,
                    'filters': str(layer.filters),
                    'kernel_size': str(layer.kernel_size),
                    'strides': str(layer.strides),
                    'activation': layer.activation.__name__
                })
            elif name == 'Dropout':
                layers.append({
                    'type': name,
                    'rate': str(layer.rate)
                })

        return {
            'layers': [
                {
                    'type': 'Input',
                    'shape': self.model.input_shape[1:]
                },
                *layers
            ],
            'optimizer': {
                'type': self.model.optimizer.__class__.__name__,
                'learning_rate': str(lr)
            },
            'loss': self.model.loss
        }
