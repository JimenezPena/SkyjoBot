"""Versión 5.0 (en desarrollo): observación y red independientes del nº de jugadores, sobre la vista pública.
Fase 2: observación + red + agente. El entrenamiento multijugador llega en la fase 3."""
VERSION = "v5_0"
from .env import SkyjoEnv, N_ACCIONES
from .observacion import N_ESCALARES, observacion_desde_vista, a_tensores
from .model import SkyjoDQN
from .agente import AgenteDQN
