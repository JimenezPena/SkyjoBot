"""Vista pública de una partida: lo ÚNICO que ve un agente (humano, modelo o heurístico).

Contiene solo información que verían los jugadores en la mesa: cartas visibles, nº de cartas ocultas por
jugador, pila de descartes, carta en mano, tamaño del mazo y quién cerró. NUNCA los valores de las cartas
ocultas ni el orden del mazo. Así los agentes de cualquier tipo y versión son intercambiables.

Orden de los tableros, relativo al asiento que decide (en el sentido del turno):
  0           = yo
  1           = izquierda: el siguiente en jugar, quien recibe mis descartes
  2 .. n-2    = resto de jugadores
  n-1         = derecha: el anterior, de quien recibo el descarte visible
Con 2 jugadores izquierda y derecha son el mismo rival (índice 1).
"""
from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass(frozen=True, eq=False)
class Vista:
    n: int                        # nº de jugadores
    yo: int                       # asiento absoluto que decide (informativo)
    valores: np.ndarray           # (n,3,4) int16: valor de las cartas visibles; 0 si oculta o eliminada
    estado: np.ndarray            # (n,3,4) int8: 1 visible, 0 oculta, -1 eliminada
    fase: int                     # 1 = robar (descarte o mazo), 2 = colocar / descartar
    mano: Optional[int]           # carta robada que hay que jugar (fase 2)
    origen: Optional[str]         # 'MAZO' | 'DESCARTE' | None
    descartes: np.ndarray         # pila de descartes en orden; la última es la visible
    n_mazo: int                   # cartas que quedan en el mazo
    cerrador: Optional[int]       # asiento relativo de quien cerró la ronda (0 = yo)
    turnos_restantes: int         # turnos que quedan tras cerrar (-1 si nadie cerró)
    mask: np.ndarray              # (26,) bool: acciones legales

    @property
    def top_descarte(self):
        return int(self.descartes[-1]) if len(self.descartes) else None

    @property
    def ocultas(self):
        """Nº de cartas ocultas por jugador (orden relativo)."""
        return (self.estado == 0).sum(axis=(1, 2))

    @property
    def puntos_visibles(self):
        return self.valores.reshape(self.n, -1).sum(axis=1)

    @property
    def izquierda(self):
        return 1

    @property
    def derecha(self):
        return self.n - 1

    def acciones_legales(self):
        return np.flatnonzero(self.mask)


def construir_vista(env, jugador_id=None):
    """Vista pública del jugador 'jugador_id' (por defecto, el que tiene el turno) a partir de un SkyjoEnv."""
    j = env.turno_actual if jugador_id is None else int(jugador_id)
    n = env.num_jugadores
    idx = [(j + i) % n for i in range(n)]
    t = env.tableros[idx]
    estado = t[:, :, :, 1].astype(np.int8)
    valores = np.where(estado == 1, t[:, :, :, 0], 0).astype(np.int16)
    descartes = np.asarray(env.descartes, dtype=np.int16)
    mask = np.array(env.get_action_mask(j), dtype=bool)
    for a in (valores, estado, descartes, mask):
        a.setflags(write=False)
    return Vista(
        n=n, yo=j, valores=valores, estado=estado, fase=int(env.fase_turno),
        mano=None if env.carta_en_mano is None else int(env.carta_en_mano),
        origen=env.origen_robo, descartes=descartes, n_mazo=len(env.mazo),
        cerrador=None if env.cerrador_id is None else (int(env.cerrador_id) - j) % n,
        turnos_restantes=int(env.turnos_restantes), mask=mask)


def entorno_desde_vista(vista, clase_env, destino=None):
    """Entorno "sombra" con SOLO la información de la vista (cartas ocultas a 0), para los modelos cuya
    observación aún se calcula desde un entorno (p. ej. v5_0). Los asientos quedan en orden relativo."""
    e = destino if destino is not None else clase_env(num_jugadores=vista.n)
    e.num_jugadores = vista.n
    e.tableros = np.stack([vista.valores, vista.estado], axis=-1).astype(e.tableros.dtype)
    e.descartes = [int(x) for x in vista.descartes]
    e.mazo = [0] * vista.n_mazo
    e.carta_en_mano = vista.mano
    e.origen_robo = vista.origen
    e.fase_turno = vista.fase
    e.turno_actual = 0
    e.cerrador_id = vista.cerrador
    e.turnos_restantes = vista.turnos_restantes
    e.game_over = False
    return e


def agente_env(agente):
    """Adapta un agente de vista a la firma antigua f(env, obs, mask) (la usan los duelos de los trainers)."""
    def f(env, obs, mask):
        return int(agente(construir_vista(env, env.turno_actual)))
    f.nombre = getattr(agente, 'nombre', 'agente')
    return f
