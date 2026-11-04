"""Versión 6.0 (en desarrollo): mismo entorno, observación y entrenamiento que v5.0, pero la red es un grafo sobre las
12 cartas del tablero (aristas = misma columna) con cabezas de acción por carta. Solo cambia model.py."""
VERSION = "v6_0"
JUGADORES = (2, 8)        # asientos que admite el modelo
from .env import SkyjoEnv, N_ACCIONES
from .observacion import N_ESCALARES, observacion_desde_vista, a_tensores
from .model import SkyjoDQN
from .agente import AgenteDQN
from .buffer import ReplayBuffer
from .agents import evaluar_vs_heuristicos, imprimir_vs_heuristicos, evaluar_final
from .trainer import entrenar_skyjo, entrenar_step
from .config import CONFIG
