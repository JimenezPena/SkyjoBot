"""Versión 5.0 (en desarrollo): observación y red independientes del nº de jugadores, sobre la vista pública,
con entrenamiento self-play de una red compartida por todos los asientos."""
VERSION = "v5_0"
from .env import SkyjoEnv, N_ACCIONES
from .observacion import N_ESCALARES, observacion_desde_vista, a_tensores
from .model import SkyjoDQN
from .agente import AgenteDQN
from .buffer import ReplayBuffer
from .agents import evaluar_vs_heuristicos, imprimir_vs_heuristicos, evaluar_final
from .trainer import entrenar_skyjo, entrenar_step
from .config import CONFIG
