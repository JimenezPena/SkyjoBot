from .env import N_ACCIONES, N_ESCALARES, N_PLANOS_PROPIO

# ============================================================
# ReplayBuffer + colector de retornos n-step por jugador (v5.0)
# ============================================================
import numpy as np
import torch
from collections import deque


class ReplayBuffer:
    def __init__(self, capacity, num_jugadores=2, device="cpu"):
        self.capacity, self.device = capacity, device
        self.pos, self.size = 0, 0
        shp = (capacity, num_jugadores, 3, 4)

        def nuevo():
            return dict(
                val=np.zeros(shp, np.float32), vis=np.zeros(shp, np.float32),
                pl=np.zeros((capacity, N_PLANOS_PROPIO, 3, 4), np.float32),
                esc=np.zeros((capacity, N_ESCALARES), np.float32),
                mask=np.zeros((capacity, N_ACCIONES), bool))
        self.o, self.n = nuevo(), nuevo()
        self.actions = np.zeros((capacity, 1), np.int64)
        self.rewards = np.zeros((capacity, 1), np.float32)
        self.gpow = np.zeros((capacity, 1), np.float32)
        self.dones = np.zeros((capacity, 1), bool)

    def _guardar(self, d, i, obs, mask):
        d['val'][i] = obs['valores_tableros']; d['vis'][i] = obs['visibilidad_tableros']
        d['pl'][i] = obs['planos_propio'];     d['esc'][i] = obs['escalares']
        d['mask'][i] = mask

    def push(self, obs, mask, action, reward, gamma_pow, next_obs, next_mask, done):
        i = self.pos
        self._guardar(self.o, i, obs, mask)
        self._guardar(self.n, i, next_obs, next_mask)
        self.actions[i], self.rewards[i], self.gpow[i], self.dones[i] = action, reward, gamma_pow, done
        self.pos = (i + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def _tensores(self, d, idx):
        t = lambda a: torch.from_numpy(a[idx]).to(self.device)
        obs = {'valores_tableros': t(d['val']), 'visibilidad_tableros': t(d['vis']),
               'planos_propio': t(d['pl']), 'escalares': t(d['esc'])}
        return obs, t(d['mask'])

    def sample(self, batch_size):
        idx = np.random.randint(0, self.size, size=batch_size)
        obs, mask = self._tensores(self.o, idx)
        nobs, nmask = self._tensores(self.n, idx)
        t = lambda a: torch.from_numpy(a[idx]).to(self.device)
        return obs, mask, t(self.actions), t(self.rewards), t(self.gpow), nobs, t(self.dones), nmask

    def __len__(self):
        return self.size


class NStepCollector:
    """Cada jugador tiene su propia cadena de decisiones (fase 1 y fase 2 cuentan como pasos).
    Emite transiciones (s_t, a_t, sum_{k<n} g^k r_{t+k}, s_{t+n}) con descuento g^n para el bootstrap."""
    def __init__(self, buffer, n, gamma):
        self.buffer, self.n, self.gamma = buffer, n, gamma
        self.pend = {}

    def reset(self):
        self.pend.clear()

    def _emit(self, items, next_obs, next_mask, done):
        ret = sum((self.gamma ** k) * it[3] for k, it in enumerate(items))
        o, m, a, _ = items[0]
        self.buffer.push(o, m, a, ret, self.gamma ** len(items), next_obs, next_mask, done)

    def observe(self, j, obs, mask):
        """Llamar al inicio de cada decisión del jugador j (antes de actuar)."""
        d = self.pend.get(j)
        if d is not None and len(d) >= self.n:
            self._emit(list(d), obs, mask, False)
            d.popleft()

    def record(self, j, obs, mask, accion, reward):
        self.pend.setdefault(j, deque()).append((obs, mask, accion, reward))

    def finish(self, terminal, final_obs):
        """Fin de partida: suma la recompensa terminal a la última decisión de cada jugador
        y vacía sus colas con done=True."""
        zero = np.zeros(N_ACCIONES, dtype=bool)
        for j, d in self.pend.items():
            items = list(d)
            if not items:
                continue
            o, m, a, r = items[-1]
            items[-1] = (o, m, a, r + terminal[j])
            for i in range(len(items)):
                self._emit(items[i:], final_obs, zero, True)
        self.pend.clear()
