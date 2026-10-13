"""Baseline heurístico (y aleatorio): un "modelo" más, sin pesos.

Solo usa información pública (su tablero, el descarte visible, la mano y las cartas ya vistas), nunca
mira las cartas ocultas, y no depende de la versión del entorno ni del número de jugadores:
funciona tal cual con cualquier SkyjoEnv (2-8 jugadores). Firma de agente: f(env, obs, mask) -> acción.
"""
import random

import numpy as np

VERSION = "heuristico"
_TOTALES = np.array([5, 10, 15] + [10] * 12, dtype=np.float32)    # nº de cartas por valor -2..12
_VALORES = np.arange(-2, 13, dtype=np.float32)


def valor_esperado_oculta(env):
    """Valor medio esperado de una carta desconocida (mazo + ocultas), según lo ya visto."""
    vistas = np.zeros(15, dtype=np.float32)
    if len(env.descartes):
        vistas += np.bincount(np.asarray(env.descartes, dtype=np.int64) + 2, minlength=15)[:15]
    for t in env.tableros:
        vis = t[:, :, 0][t[:, :, 1] == 1]
        if vis.size:
            vistas += np.bincount(vis.astype(np.int64) + 2, minlength=15)[:15]
    if env.carta_en_mano is not None:
        vistas[int(env.carta_en_mano) + 2] += 1.0
    conteo = np.maximum(_TOTALES - vistas, 0.0)
    total = float(conteo.sum())
    return 5.0 if total <= 0 else float((conteo * _VALORES).sum() / total)


def _ganancias(env, j, carta, e_oculta):
    """Puntos que ahorra colocar 'carta' en cada casilla [3,4] del jugador j (-1e9 si no es posible).
    Si completa una columna, esta desaparece entera."""
    t = env.tableros[j]
    gan = np.full((3, 4), -1e9, dtype=np.float32)
    for c in range(4):
        vis, vals = t[:, c, 1], t[:, c, 0]
        if vis[0] == -1:                      # columna ya eliminada
            continue
        for r in range(3):
            old = float(vals[r]) if vis[r] == 1 else e_oculta
            otros = [k for k in range(3) if k != r]
            if all(vis[k] == 1 and vals[k] == carta for k in otros):
                gan[r, c] = 2 * carta + old
            else:
                gan[r, c] = old - carta
    return gan


def agente_aleatorio(env, obs, mask):
    return int(random.choice(np.flatnonzero(mask)))


def agente_heuristico(env, obs, mask):
    """- coge el descarte si mejora >= 3 pts esperados (o completa columna)
    - coloca la carta donde más gana; si es del mazo y gana < 2, la descarta y destapa una oculta."""
    j = env.turno_actual
    e = valor_esperado_oculta(env)
    if env.fase_turno == 1:
        gan = _ganancias(env, j, int(env.descartes[-1]), e)
        return 0 if gan.max() >= 3 else 1
    gan = _ganancias(env, j, int(env.carta_en_mano), e)
    mejor = int(np.argmax(gan))                       # índice plano f*4+c = acción - 2
    ocultas = np.flatnonzero(mask[14:])
    if env.origen_robo == 'MAZO' and gan.flat[mejor] < 2 and len(ocultas) > 0:
        return 14 + int(random.choice(ocultas))
    return 2 + mejor
