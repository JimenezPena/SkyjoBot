import torch
import random
import numpy as np

import torch.nn as nn
import torch.nn.functional as F

from skyjo_env import N_ESCALARES, N_ACCIONES

class SkyjoDQN(nn.Module):
    """
    Red Neuronal Q-Network para Skyjo.
    Entradas:
      - Tableros de los jugadores (Valores y Visibilidad).
      - Información escalar (Carta en mano, fase del turno, origen de robo).
    Salida:
      - 26 Q-values (uno por cada acción posible).
    """
    def __init__(self, num_jugadores=2):
        super(SkyjoDQN, self).__init__()
        self.num_jugadores = num_jugadores

        # ------------------------------------------------------------------
        # 1. Extractor de características espaciales del tablero (CNN)
        # ------------------------------------------------------------------
        # Entrada por tablero: [Batch, 2 canales (valores, visibilidad), 3 filas, 4 columnas]
        self.conv1 = nn.Conv2d(in_channels=2, out_channels=32, kernel_size=(3, 3), padding=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(2, 2), padding=0)

        # Dimensión a la salida de las Convoluciones por cada tablero:
        # Tras conv1 (padding=1): 3x4 -> conv2 (kernel=2x2, stride=1): 2x3 con 64 canales = 64 * 2 * 3 = 384
        flatten_size_per_board = 64 * 2 * 3
        total_boards_feature_size = flatten_size_per_board * num_jugadores

        # ------------------------------------------------------------------
        # 2. Extractor de características escalares (Información del turno)
        # ------------------------------------------------------------------
        # Entrada escalar: [carta_en_mano, top_descarte, fase_turno, origen_robo_flag + 15 de conteo de cartas] 
        #                            [+ 9 features de contexto -> 28 características:  (N_ESCALARES)                                 ]
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
        Paso hacia adelante (Forward Pass).
        obs_dict es un diccionario o tupla con:
          - 'valores_tableros': Tensor [Batch, Num_Jugadores, 3, 4]
          - 'visibilidad_tableros': Tensor [Batch, Num_Jugadores, 3, 4]
          - 'escalares': Tensor [Batch, N_ESCALARES=28] -> (carta_en_mano, top_descartes, fase, origen_robo + 15 conteo)
        """
        val_tableros = obs_dict['valores_tableros']       # Shape: [B, N, 3, 4]
        vis_tableros = obs_dict['visibilidad_tableros']   # Shape: [B, N, 3, 4]
        escalares = obs_dict['escalares']                 # Shape: [B, 3]

        batch_size = val_tableros.size(0)
        board_features = []

        # Procesar el tablero de cada jugador a través del extractor CNN compartida
        for j in range(self.num_jugadores):
            # Concatenar canal de valores y visibilidad -> Shape: [B, 2, 3, 4]
            tablero_j = torch.stack([val_tableros[:, j], vis_tableros[:, j]], dim=1)

            x = F.relu(self.conv1(tablero_j))
            x = F.relu(self.conv2(x))
            x = x.view(batch_size, -1)  # Aplanar a vector [B, 384]
            board_features.append(x)

        # Concatenar las características de todos los tableros -> [B, N * 384]
        boards_combined = torch.cat(board_features, dim=1)

        # Procesar variables escalares -> [B, 64]
        scalar_features = self.fc_scalars(escalares)

        # Unir características de tableros y variables de turno -> [B, N*384 + 32]
        merged = torch.cat([boards_combined, scalar_features], dim=1)

        # Capas totalmente conectadas
        h = F.relu(self.fc1(merged))
        h = F.relu(self.fc2(h))

        # Emitir los 26 Q-Values
        #out_q_values = self.q_values(h)
        
        # Cálculo de las ramas Dueling
        values = self.value_stream(h)        # Shape: [B, 1]
        advantages = self.advantage_stream(h) # Shape: [B, N_ACCIONES]

        # Combinación centrada en la media: Q(s,a) = V(s) + (A(s,a) - mean(A(s,a')))
        out_q_values = values + (advantages - advantages.mean(dim=1, keepdim=True))

        return out_q_values
        

def seleccionar_accion_dqn(model, obs, mask, device="cpu"):
    """
    Selección de acción en modo 100% explotación (Epsilon = 0).
    Aplica Action Masking riguroso.
    """
    model.eval()
    with torch.no_grad():
        obs_tensor = {
            'valores_tableros': torch.FloatTensor(obs['valores_tableros']).unsqueeze(0).to(device),
            'visibilidad_tableros': torch.FloatTensor(obs['visibilidad_tableros']).unsqueeze(0).to(device),
            'escalares': torch.FloatTensor(obs['escalares']).unsqueeze(0).to(device)
        }

        q_values = model(obs_tensor)

    # Filtrar acciones ilegales (-1e9 a lo que esté en False)
    mask_tensor = torch.BoolTensor(mask).to(device)
    q_values[0, ~mask_tensor] = -1e9

    return torch.argmax(q_values, dim=1).item()

