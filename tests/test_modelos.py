import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

import skyjo.v4_2 as v4
import skyjo.v5_0 as v5
from skyjo.heuristico import agente_aleatorio, agente_heuristico


def _partida(mod, seed):
    env = mod.SkyjoEnv(2); random.seed(seed); env.reset()
    rng, n, done = random.Random(seed + 1000), 0, False
    while not done:
        _, _, done, _ = env.step(rng.choice(np.flatnonzero(env.get_action_mask()).tolist())); n += 1
    return n, env.calcular_puntuaciones_finales()


def test_mismas_reglas_en_todas_las_versiones():
    for s in range(40):
        assert _partida(v4, s) == _partida(v5, s), s


def test_heuristico_juega_con_cualquier_numero_de_jugadores():
    for mod in (v4, v5):
        for n in (2, 3, 4, 6):
            env = mod.SkyjoEnv(n)
            for _ in range(3):
                env.reset(); done, k = False, 0
                while not done and k < 5000:
                    m = env.get_action_mask()
                    _, _, done, _ = env.step(agente_heuristico(env, None, m)); k += 1
                assert done and len(env.calcular_puntuaciones_finales()) == n


def test_heuristico_supera_al_aleatorio():
    env, vic, N = v4.SkyjoEnv(2), 0, 60
    for p in range(N):
        env.reset(); done = False
        while not done:
            m = env.get_action_mask()
            a = agente_heuristico(env, None, m) if env.turno_actual == p % 2 else agente_aleatorio(env, None, m)
            _, _, done, _ = env.step(a)
        pts = env.calcular_puntuaciones_finales()
        vic += pts[p % 2] < pts[1 - p % 2]
    assert vic / N > 0.8


if __name__ == "__main__":
    for k, f in list(globals().items()):
        if k.startswith("test_"):
            f(); print("ok", k)
