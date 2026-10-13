"""Agente DQN 5.0 sobre la vista pública. La observación 5.0 aún se calcula desde un entorno, así que se
construye un entorno "sombra" solo con la información de la vista (transitorio hasta portarla)."""
import threading

import numpy as np

from skyjo.agentes import Agente
from skyjo.vista import entorno_desde_vista
from .env import SkyjoEnv
from .acciones import accion_greedy


class AgenteDQN(Agente):
    def __init__(self, model, device="cpu", nombre="dqn_v5_0"):
        self.model, self.device, self.nombre = model.eval(), device, nombre
        self._sombra = SkyjoEnv(num_jugadores=model.num_jugadores)
        self._lock = threading.Lock()                 # la web puede atender varias partidas a la vez

    def decidir(self, vista):
        if vista.n != self.model.num_jugadores:
            raise ValueError(f"Este modelo solo juega con {self.model.num_jugadores} jugadores (la vista tiene {vista.n}).")
        with self._lock:
            obs = entorno_desde_vista(vista, SkyjoEnv, destino=self._sombra).obtener_observacion(0)
        return accion_greedy(self.model, obs, np.array(vista.mask), self.device)
