"""Árbitro entre versiones.

Las reglas del juego son idénticas en todas las versiones; lo que cambia es la observación y la red.
Un agente de la versión X mantiene un entorno "sombra" de su versión: antes de decidir copia el estado
de la partida del entorno árbitro, calcula SU observación y decide con SU red.
Así se pueden enfrentar redes (o la web puede mezclar niveles) de versiones distintas.
Todos los agentes comparten la firma f(env, obs, mask) -> acción.
"""
import threading

import numpy as np

_ATRIBUTOS = ('carta_en_mano', 'origen_robo', 'fase_turno', 'turno_actual',
              'cerrador_id', 'turnos_restantes', 'game_over')


def sincronizar(origen, destino):
    """Copia al entorno 'destino' el estado de la partida de 'origen' (sin tocar nada de 'origen')."""
    destino.tableros = np.array(origen.tableros, dtype=destino.tableros.dtype)
    destino.num_jugadores = origen.num_jugadores
    destino.descartes = list(origen.descartes)
    destino.mazo = origen.mazo            # solo se lee (len); los agentes no roban cartas
    for a in _ATRIBUTOS:
        setattr(destino, a, getattr(origen, a))


class AgenteSincronizado:
    """Adapta una política de la versión 'modulo' (p. ej. skyjo.v4_2) a cualquier entorno árbitro."""
    def __init__(self, modulo, politica):
        self.env = modulo.SkyjoEnv(num_jugadores=2)
        self.politica = politica
        self._lock = threading.Lock()      # la web puede atender varias partidas a la vez

    def __call__(self, env, obs, mask):
        with self._lock:
            sincronizar(env, self.env)
            obs_v = self.env.obtener_observacion(env.turno_actual)
            return int(self.politica(self.env, obs_v, mask))
