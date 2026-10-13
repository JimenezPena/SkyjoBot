from .env import N_ESCALARES, N_ACCIONES

import torch
import torch.nn as nn
import torch.nn.functional as F

class SkyjoDQN(nn.Module):
    """
    Red Neuronal Dueling DQN para Skyjo (v4.1).
    Entradas:
      - Tableros de los jugadores (Valores y Visibilidad).
      - Información escalar (Carta en mano, fase del turno, origen de robo, conteo y contexto).
    Salida:
      - 26 Q-values combinados a partir de las ramas independientes de Valor V(s) y Ventaja A(s, a).
    """
    def __init__(self, num_jugadores=2):
        super(SkyjoDQN, self).__init__()
        self.num_jugadores = num_jugadores

        # ------------------------------------------------------------------
        # 1. Extractor de características espaciales del tablero (CNN)
        # ------------------------------------------------------------------
        self.conv1 = nn.Conv2d(in_channels=2, out_channels=32, kernel_size=(3, 3), padding=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(2, 2), padding=0)

        flatten_size_per_board = 64 * 2 * 3
        total_boards_feature_size = flatten_size_per_board * num_jugadores

        # ------------------------------------------------------------------
        # 2. Extractor de características escalares
        # ------------------------------------------------------------------
        self.fc_scalars = nn.Sequential(
            nn.Linear(N_ESCALARES, 64),
            nn.ReLU()
        )

        # ------------------------------------------------------------------
        # 3. Representación Oculta Común
        # ------------------------------------------------------------------
        combined_size = total_boards_feature_size + 64

        self.fc1 = nn.Linear(combined_size, 256)
        self.fc2 = nn.Linear(256, 128)

        # ------------------------------------------------------------------
        # 4. Arquitectura Dueling DQN (Cabezas Separadas)
        # ------------------------------------------------------------------
        # Stream de Valor V(s): Estima la calidad del estado global [Batch, 1]
        self.value_stream = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

        # Stream de Ventaja A(s, a): Estima la ventaja relativa de cada acción [Batch, 26]
        self.advantage_stream = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, N_ACCIONES)
        )

    def forward(self, obs_dict):
        """
        Paso hacia adelante (Forward Pass) con Dueling DQN aggregation.
        """
        val_tableros = obs_dict['valores_tableros']       # Shape: [B, N, 3, 4]
        vis_tableros = obs_dict['visibilidad_tableros']   # Shape: [B, N, 3, 4]
        escalares = obs_dict['escalares']                 # Shape: [B, 28]

        batch_size = val_tableros.size(0)
        board_features = []

        # Extractor CNN compartido
        for j in range(self.num_jugadores):
            tablero_j = torch.stack([val_tableros[:, j], vis_tableros[:, j]], dim=1)
            x = F.relu(self.conv1(tablero_j))
            x = F.relu(self.conv2(x))
            x = x.view(batch_size, -1)
            board_features.append(x)

        boards_combined = torch.cat(board_features, dim=1)
        scalar_features = self.fc_scalars(escalares)
        merged = torch.cat([boards_combined, scalar_features], dim=1)

        # Representación profunda común
        h = F.relu(self.fc1(merged))
        h = F.relu(self.fc2(h))

        # Cálculo de las ramas Dueling
        values = self.value_stream(h)        # Shape: [B, 1]
        advantages = self.advantage_stream(h) # Shape: [B, N_ACCIONES]

        # Combinación centrada en la media: Q(s,a) = V(s) + (A(s,a) - mean(A(s,a')))
        out_q_values = values + (advantages - advantages.mean(dim=1, keepdim=True))
        return out_q_values
