"""Versión 5.0: observación con columnas, dueling enmascarado, n-step y soft update."""
VERSION = "v5_0"
from .env import SkyjoEnv, N_ACCIONES, N_ESCALARES, N_PLANOS_PROPIO
from .model import SkyjoDQN
from .buffer import ReplayBuffer, NStepCollector
from .acciones import seleccionar_accion, accion_greedy
from .agents import (agente_aleatorio, agente_heuristico, agente_dqn,
                     duelo, imprimir_duelo)
from .trainer import entrenar_skyjo, entrenar_step
from .config import CONFIG
