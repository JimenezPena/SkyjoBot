"""Versión 4.2: Dueling Double DQN (modelo idéntico a 4.1) + bot heurístico y duelos.
Los pesos de 4.1 (skyjo_dqn_model_4.1.pth) se cargan con esta misma red."""
VERSION = "v4_2"
from .env import SkyjoEnv, N_ACCIONES, N_ESCALARES
from .model import SkyjoDQN
from .observacion import observacion_desde_vista
from .agente import AgenteDQN
from .buffer import ReplayBuffer
from .acciones import seleccionar_accion
from .agents import (agente_aleatorio, agente_heuristico, agente_dqn, accion_greedy,
                     duelo, imprimir_duelo)
from .trainer import entrenar_skyjo, entrenar_step
from .config import CONFIG
