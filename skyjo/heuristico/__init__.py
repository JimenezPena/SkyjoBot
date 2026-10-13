"""Baseline heurístico (y aleatorio): un "modelo" más, sin pesos.

Solo usa la vista pública, así que funciona con cualquier versión del entorno y con 2-8 jugadores.
"""
import random

import numpy as np

from skyjo.agentes import Agente

VERSION = "heuristico"
_TOTALES = np.array([5, 10, 15] + [10] * 12, dtype=np.float32)    # nº de cartas por valor -2..12
_VALORES = np.arange(-2, 13, dtype=np.float32)


def valor_esperado_oculta(vista):
    """Valor medio esperado de una carta desconocida (mazo + ocultas), según lo ya visto."""
    vistas = np.zeros(15, dtype=np.float32)
    if len(vista.descartes):
        vistas += np.bincount(vista.descartes.astype(np.int64) + 2, minlength=15)[:15]
    vis = vista.valores[vista.estado == 1]
    if vis.size:
        vistas += np.bincount(vis.astype(np.int64) + 2, minlength=15)[:15]
    if vista.mano is not None:
        vistas[vista.mano + 2] += 1.0
    conteo = np.maximum(_TOTALES - vistas, 0.0)
    total = float(conteo.sum())
    return 5.0 if total <= 0 else float((conteo * _VALORES).sum() / total)


def _ganancias(vista, carta, e_oculta):
    """Puntos que ahorra colocar 'carta' en cada casilla [3,4] de MI tablero (-1e9 si no es posible).
    Si completa una columna, esta desaparece entera."""
    vals, est = vista.valores[0], vista.estado[0]
    gan = np.full((3, 4), -1e9, dtype=np.float32)
    for c in range(4):
        vis, v = est[:, c], vals[:, c]
        if vis[0] == -1:                      # columna ya eliminada
            continue
        for r in range(3):
            old = float(v[r]) if vis[r] == 1 else e_oculta
            otros = [k for k in range(3) if k != r]
            if all(vis[k] == 1 and v[k] == carta for k in otros):
                gan[r, c] = 2 * carta + old
            else:
                gan[r, c] = old - carta
    return gan


class AgenteAleatorio(Agente):
    nombre = "aleatorio"

    def decidir(self, vista):
        return int(random.choice(vista.acciones_legales()))


class AgenteHeuristico(Agente):
    """- coge el descarte si mejora >= 3 pts esperados (o completa columna)
    - coloca la carta donde más gana; si es del mazo y gana < 2, la descarta y destapa una oculta."""
    nombre = "heuristico"

    def decidir(self, vista):
        e = valor_esperado_oculta(vista)
        if vista.fase == 1:
            gan = _ganancias(vista, vista.top_descarte, e)
            return 0 if gan.max() >= 3 else 1
        gan = _ganancias(vista, vista.mano, e)
        mejor = int(np.argmax(gan))                       # índice plano f*4+c = acción - 2
        ocultas = np.flatnonzero(vista.mask[14:])
        if vista.origen == 'MAZO' and gan.flat[mejor] < 2 and len(ocultas) > 0:
            return 14 + int(random.choice(ocultas))
        return 2 + mejor
