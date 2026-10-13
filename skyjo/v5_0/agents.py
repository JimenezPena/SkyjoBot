from .env import SkyjoEnv
from .acciones import accion_greedy

# ============================================================
# Evaluación: bot heurístico burdo + duelos entre agentes (v5.0)
# Un agente es una función f(env, obs, mask) -> acción
# ============================================================
import random
import numpy as np


from skyjo.heuristico import agente_aleatorio, agente_heuristico   # baseline compartido


def agente_dqn(model, device="cpu"):
    def _a(env, obs, mask):
        return accion_greedy(model, obs, mask, device)
    return _a


def duelo(agente_a, agente_b, n_partidas=200, max_pasos=2000):
    """A vs B alternando asiento. Devuelve métricas desde el punto de vista de A."""
    env = SkyjoEnv(num_jugadores=2)
    env.metricas = True
    r = dict(n=n_partidas, trunc=0, vic=0, emp=0, pts_a=0.0, pts_b=0.0, cols_a=0, cols_b=0, reg_a=0, reg_b=0)
    for p in range(n_partidas):
        a_es_0 = (p % 2 == 0)
        obs, done, pasos = env.reset(), False, 0
        while not done and pasos < max_pasos:     # tope: una red degenerada puede no cerrar nunca la partida
            pasos += 1
            j = env.turno_actual
            mask = env.get_action_mask(j)
            ag = agente_a if ((j == 0) == a_es_0) else agente_b
            obs, _, done, _ = env.step(ag(env, obs, mask))
        r['trunc'] += (not done)
        puntos = env.calcular_puntuaciones_finales()
        ia, ib = (0, 1) if a_es_0 else (1, 0)
        r['vic'] += puntos[ia] < puntos[ib]
        r['emp'] += puntos[ia] == puntos[ib]
        r['pts_a'] += puntos[ia]; r['pts_b'] += puntos[ib]
        r['cols_a'] += env.columnas_eliminadas[ia]; r['cols_b'] += env.columnas_eliminadas[ib]
        r['reg_a'] += env.regalos[ia];              r['reg_b'] += env.regalos[ib]
    r['winrate'] = (r['vic'] + 0.5 * r['emp']) / n_partidas
    return r


def imprimir_duelo(nombre, r):
    n = r['n']
    print(f"{nombre:<28} win {100*r['vic']/n:5.1f}% (emp {100*r['emp']/n:4.1f}%) | "
          f"pts A/B {r['pts_a']/n:5.1f}/{r['pts_b']/n:5.1f} | "
          f"cols/partida A/B {r['cols_a']/n:.2f}/{r['cols_b']/n:.2f} | "
          f"regalos/partida A/B {r['reg_a']/n:.2f}/{r['reg_b']/n:.2f}")
    if r.get('trunc'):
        print(f"   ({r['trunc']} partidas truncadas por exceder el máximo de pasos)")
