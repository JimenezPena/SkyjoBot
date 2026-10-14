import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

import skyjo.v4_2 as v4
import skyjo.v5_0 as v5
from skyjo.heuristico import AgenteAleatorio, AgenteHeuristico
from skyjo.partida import evaluar
from skyjo.vista import construir_vista


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
    ag = AgenteHeuristico()
    for mod in (v4, v5):
        for n in (2, 3, 4, 6):
            env = mod.SkyjoEnv(n)
            for _ in range(3):
                env.reset(); done, k = False, 0
                while not done and k < 5000:
                    _, _, done, _ = env.step(ag(construir_vista(env))); k += 1
                assert done and len(env.calcular_puntuaciones_finales()) == n


def test_heuristico_supera_al_aleatorio():
    res = evaluar([AgenteHeuristico(), AgenteAleatorio()], n_partidas=60)
    assert res["heuristico"]["winrate"] > 0.8
