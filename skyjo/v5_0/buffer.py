import numpy as np
import torch

from .env import N_ACCIONES


class ReplayBuffer:
    """Replay buffer para observaciones tipo diccionario (cualquier nº de jugadores fijo por buffer).
    Las formas de cada campo se toman de una observación de ejemplo."""
    def __init__(self, capacity, ejemplo, device="cpu"):
        self.capacity, self.device, self.ptr, self.size = capacity, device, 0, 0
        self.obs = {k: np.zeros((capacity, *np.shape(v)), dtype=np.float32) for k, v in ejemplo.items()}
        self.next_obs = {k: np.zeros((capacity, *np.shape(v)), dtype=np.float32) for k, v in ejemplo.items()}
        self.actions = np.zeros((capacity, 1), dtype=np.int64)
        self.rewards = np.zeros((capacity, 1), dtype=np.float32)
        self.dones = np.zeros((capacity, 1), dtype=np.bool_)
        self.next_masks = np.zeros((capacity, N_ACCIONES), dtype=np.bool_)

    def push(self, obs, action, reward, next_obs, done, next_mask):
        for k in self.obs:
            self.obs[k][self.ptr] = obs[k]
            self.next_obs[k][self.ptr] = next_obs[k]
        self.actions[self.ptr], self.rewards[self.ptr], self.dones[self.ptr] = action, reward, done
        self.next_masks[self.ptr] = next_mask
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size):
        i = np.random.randint(0, self.size, size=batch_size)
        t = lambda a, tipo=torch.FloatTensor: tipo(a[i]).to(self.device)
        return ({k: t(v) for k, v in self.obs.items()}, t(self.actions, torch.LongTensor), t(self.rewards),
                {k: t(v) for k, v in self.next_obs.items()}, t(self.dones, torch.BoolTensor),
                t(self.next_masks, torch.BoolTensor))

    def __len__(self):
        return self.size
