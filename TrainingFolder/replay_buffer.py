import random
import numpy as np
import torch

class ReplayBuffer:
    """
    Buffer de Replay de Experiencia adaptado a observaciones tipo Diccionario y Action Masks.
    """
    def __init__(self, capacity, num_jugadores=2, device="cpu"):
        self.capacity = capacity
        self.device = device
        self.num_jugadores = num_jugadores
        self.ptr = 0
        self.size = 0

        # ------------------------------------------------------------------
        # Reservar memoria fija en NumPy para máximo rendimiento
        # ------------------------------------------------------------------
        # Estado Actual (S)
        self.obs_val = np.zeros((capacity, num_jugadores, 3, 4), dtype=np.float32)
        self.obs_vis = np.zeros((capacity, num_jugadores, 3, 4), dtype=np.float32)
        self.obs_esc = np.zeros((capacity, N_ESCALARES), dtype=np.float32)

        # Transiciones (A, R, Done)
        self.actions = np.zeros((capacity, 1), dtype=np.int64)
        self.rewards = np.zeros((capacity, 1), dtype=np.float32)
        self.dones = np.zeros((capacity, 1), dtype=np.bool_)

        # Estado Siguiente (S')
        self.next_obs_val = np.zeros((capacity, num_jugadores, 3, 4), dtype=np.float32)
        self.next_obs_vis = np.zeros((capacity, num_jugadores, 3, 4), dtype=np.float32)
        self.next_obs_esc = np.zeros((capacity, N_ESCALARES), dtype=np.float32)

        # Máscara de acciones legales para S'
        self.next_masks = np.zeros((capacity, 15), dtype=np.bool_)

    def push(self, obs, action, reward, next_obs, done, next_mask):
        """
        Guarda una transición individual en el buffer en formato circular.
        """
        # Guardar Estado Actual (obs)
        self.obs_val[self.ptr] = obs['valores_tableros']
        self.obs_vis[self.ptr] = obs['visibilidad_tableros']
        self.obs_esc[self.ptr] = obs['escalares']

        # Guardar Acción, Recompensa y Done
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.dones[self.ptr] = done

        # Guardar Estado Siguiente (next_obs) y su Máscara
        self.next_obs_val[self.ptr] = next_obs['valores_tableros']
        self.next_obs_vis[self.ptr] = next_obs['visibilidad_tableros']
        self.next_obs_esc[self.ptr] = next_obs['escalares']
        self.next_masks[self.ptr] = next_mask

        # Actualizar punteros circulares
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size):
        """
        Muestra un lote aleatorio de transiciones y las devuelve como Tensores de PyTorch.
        """
        indices = np.random.randint(0, self.size, size=batch_size)

        # Reconstruir diccionarios de observación como Tensores de PyTorch
        obs_batch = {
            'valores_tableros': torch.FloatTensor(self.obs_val[indices]).to(self.device),
            'visibilidad_tableros': torch.FloatTensor(self.obs_vis[indices]).to(self.device),
            'escalares': torch.FloatTensor(self.obs_esc[indices]).to(self.device)
        }

        next_obs_batch = {
            'valores_tableros': torch.FloatTensor(self.next_obs_val[indices]).to(self.device),
            'visibilidad_tableros': torch.FloatTensor(self.next_obs_vis[indices]).to(self.device),
            'escalares': torch.FloatTensor(self.next_obs_esc[indices]).to(self.device)
        }

        actions_tensor = torch.LongTensor(self.actions[indices]).to(self.device)
        rewards_tensor = torch.FloatTensor(self.rewards[indices]).to(self.device)
        dones_tensor = torch.BoolTensor(self.dones[indices]).to(self.device)
        next_masks_tensor = torch.BoolTensor(self.next_masks[indices]).to(self.device)

        return obs_batch, actions_tensor, rewards_tensor, next_obs_batch, dones_tensor, next_masks_tensor

    def __len__(self):
        return self.size
