import numpy as np
from collections import deque


class Buffer:
    def __init__(self, max_size):
        self.max_size = max_size
        self.buffer = deque(maxlen=max_size)

    def append(self, item):
        self.buffer.append(item)

    def __setitem__(self, idx, item):
        if idx < 0 or idx >= len(self.buffer):
            raise IndexError

        self.buffer[idx] = item

    def __getitem__(self, idx):
        if idx < 0 or idx >= len(self.buffer):
            raise IndexError

        return self.buffer[idx]

    def __len__(self):
        return len(self.buffer)


class Memory:
    def __init__(self, max_size: int) -> None:
        self.max_size = max_size
        self.memory = Buffer(max_size)
        self.error = Buffer(max_size)

    def append_memory(self, item) -> None:
        self.memory.append(item)

    def append_error(self, error) -> None:
        self.error.append(error)

    def set_error(self, idx, error) -> None:
        self.error[idx] = error

    def sample(self, batch_size: int) -> list:
        raise NotImplementedError

    def prioritized_sample(self, batch_size: int) -> list:
        raise NotImplementedError

    def config(self) -> dict:
        return {
            'memory_size': self.max_size,
            'last_memory_size': len(self.memory),
        }

    def __len__(self) -> int:
        return len(self.memory)


class SequentialMemory(Memory):
    def __init__(self, max_size: int) -> None:
        super().__init__(max_size)

    def sample(self, batch_size: int) -> list:
        idx = np.random.choice(len(self.memory), batch_size, replace=False)
        return [self.memory[i] for i in idx]

    def prioritized_sample(self, batch_size: int, alpha: float,
                           beta: float) -> list:
        p = np.array([abs(e) + 1e-6 for e in self.error.buffer]) ** alpha
        p = p / np.sum(p)

        idx = np.random.choice(len(self.memory), batch_size, p=p,
                               replace=False)

        p = p[idx]

        w = (len(self.memory) * p) ** (-beta)
        w = w / np.max(w)

        return [(i, *self.memory[i], w[k]) for k, i in enumerate(idx)]
