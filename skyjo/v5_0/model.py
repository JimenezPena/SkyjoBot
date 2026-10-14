import torch
import torch.nn as nn
import torch.nn.functional as F

from .env import N_ACCIONES
from .observacion import N_ESCALARES

D = 64                                   # tamaño del embedding de cada tablero


class SkyjoDQN(nn.Module):
    """Dueling DQN para cualquier nº de jugadores (v5.0). Los parámetros NO dependen de n.

    Un codificador convolucional COMPARTIDO convierte cada tablero 3x4 en un vector; cada rol tiene su propia
    proyección (yo, izquierda = quien recibe mis descartes, derecha = de quien recibo el descarte) y el resto de
    rivales se proyectan igual y se resumen con media y máximo (invariante al orden y al número).
    """
    def __init__(self, num_jugadores=None):
        super().__init__()
        self.num_jugadores = None        # None = cualquier nº (2-8); el argumento existe por compatibilidad con niveles.py
        self.conv1 = nn.Conv2d(2, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=2)
        flat = 64 * 2 * 3
        self.p_yo, self.p_izq, self.p_der, self.p_otros = (nn.Linear(flat, D) for _ in range(4))
        self.fc_escalares = nn.Sequential(nn.Linear(N_ESCALARES, 64), nn.ReLU())
        self.fc1 = nn.Linear(D * 5 + 64, 256)
        self.fc2 = nn.Linear(256, 128)
        self.value_stream = nn.Sequential(nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 1))
        self.advantage_stream = nn.Sequential(nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, N_ACCIONES))

    def _codificar(self, x):             # x: [M, 2, 3, 4] -> [M, 384]
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        return x.flatten(1)

    def forward(self, obs):
        B = obs['escalares'].size(0)
        yo = F.relu(self.p_yo(self._codificar(obs['tablero_yo'])))
        izq = F.relu(self.p_izq(self._codificar(obs['tablero_izq'])))
        der = F.relu(self.p_der(self._codificar(obs['tablero_der'])))
        otros = obs['tableros_otros']                           # [B, k, 2, 3, 4], k puede ser 0
        k = otros.size(1)
        if k:
            e = F.relu(self.p_otros(self._codificar(otros.reshape(B * k, *otros.shape[2:])))).view(B, k, D)
            pool = torch.cat([e.mean(dim=1), e.max(dim=1).values], dim=1)
        else:
            pool = yo.new_zeros(B, 2 * D)
        esc = self.fc_escalares(obs['escalares'])
        h = F.relu(self.fc1(torch.cat([yo, izq, der, pool, esc], dim=1)))
        h = F.relu(self.fc2(h))
        v, a = self.value_stream(h), self.advantage_stream(h)
        return v + (a - a.mean(dim=1, keepdim=True))
