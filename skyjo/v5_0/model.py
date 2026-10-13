from .env import N_ACCIONES, N_ESCALARES, N_PLANOS_PROPIO

# ============================================================
# Modelo: Dueling DQN con ventaja centrada SOLO sobre acciones legales (v5.0)
# ============================================================
import torch
import torch.nn as nn
import torch.nn.functional as F


class SkyjoDQN(nn.Module):
    def __init__(self, num_jugadores=2, n_acciones=N_ACCIONES, n_escalares=N_ESCALARES,
                 n_planos_propio=N_PLANOS_PROPIO):
        super().__init__()
        self.num_jugadores = num_jugadores
        # Tablero propio: valor, visibilidad + planos de columna (completa / ayuda rival / ganancia)
        self.own1 = nn.Conv2d(2 + n_planos_propio, 32, kernel_size=3, padding=1)
        self.own2 = nn.Conv2d(32, 64, kernel_size=2)
        # Tableros rivales (pesos compartidos entre rivales): valor, visibilidad
        self.riv1 = nn.Conv2d(2, 32, kernel_size=3, padding=1)
        self.riv2 = nn.Conv2d(32, 64, kernel_size=2)
        flat = 64 * 2 * 3                                   # (3x4) -> (2x3) tras la 2ª conv
        self.fc_esc = nn.Linear(n_escalares, 128)
        self.fc1 = nn.Linear(flat * num_jugadores + 128, 256)
        self.fc2 = nn.Linear(256, 128)
        self.value = nn.Linear(128, 1)
        self.adv = nn.Linear(128, n_acciones)

    def forward(self, obs, mask):
        val = obs['valores_tableros']          # [B, N, 3, 4]
        vis = obs['visibilidad_tableros']      # [B, N, 3, 4]
        own = torch.cat([val[:, :1], vis[:, :1], obs['planos_propio']], dim=1)
        x = [F.relu(self.own2(F.relu(self.own1(own)))).flatten(1)]
        for j in range(1, self.num_jugadores):
            r = torch.stack([val[:, j], vis[:, j]], dim=1)
            x.append(F.relu(self.riv2(F.relu(self.riv1(r)))).flatten(1))
        x.append(F.relu(self.fc_esc(obs['escalares'])))
        h = F.relu(self.fc1(torch.cat(x, dim=1)))
        h = F.relu(self.fc2(h))

        v = self.value(h)                      # [B, 1]
        a = self.adv(h)                        # [B, A]
        m = mask.float()
        a_mean = (a * m).sum(dim=1, keepdim=True) / m.sum(dim=1, keepdim=True).clamp(min=1.0)
        return v + a - a_mean
