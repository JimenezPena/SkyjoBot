"""v5.0: observación y red independientes del nº de jugadores, construidas solo desde la vista."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

from skyjo.heuristico import AgenteHeuristico, AgenteAleatorio
from skyjo.partida import evaluar
from skyjo.vista import construir_vista
from skyjo.v5_0 import (SkyjoDQN, AgenteDQN, SkyjoEnv, N_ESCALARES, N_ACCIONES,
                        observacion_desde_vista, a_tensores)


def _vistas(n, pasos=40, semilla=0):
    rng = np.random.default_rng(semilla)
    env = SkyjoEnv(num_jugadores=n); env.reset()
    out = []
    for _ in range(pasos):
        if env.game_over:
            break
        out.append((env, construir_vista(env)))
        env.step(int(rng.choice(np.flatnonzero(env.get_action_mask(env.turno_actual)))))
    return out


def test_formas_para_2_a_8_jugadores_y_parametros_independientes_de_n():
    torch.manual_seed(0)
    m = SkyjoDQN()
    for n in range(2, 9):
        obs = [observacion_desde_vista(v) for _, v in _vistas(n, 6)]
        assert obs[0]['escalares'].shape == (N_ESCALARES,)
        assert obs[0]['tableros_otros'].shape == (max(n - 3, 0), 2, 3, 4)
        with torch.no_grad():
            q = m(a_tensores(obs))
        assert q.shape == (len(obs), N_ACCIONES) and torch.isfinite(q).all()
    assert SkyjoDQN(2).state_dict().keys() == SkyjoDQN(8).state_dict().keys()


def test_la_observacion_no_depende_de_las_cartas_ocultas():
    env = SkyjoEnv(num_jugadores=4); env.reset()
    for _ in range(5):
        env.step(int(np.flatnonzero(env.get_action_mask(env.turno_actual))[0]))
    o1 = observacion_desde_vista(construir_vista(env, 1))
    t = env.tableros.copy()
    ocultas = t[:, :, :, 1] == 0
    t[:, :, :, 0] = np.where(ocultas, 7, t[:, :, :, 0])          # otras cartas ocultas
    env.tableros = t
    o2 = observacion_desde_vista(construir_vista(env, 1))
    for k in o1:
        assert np.array_equal(o1[k], o2[k]), k


def test_la_red_es_invariante_al_orden_de_los_otros_rivales():
    torch.manual_seed(1)
    m = SkyjoDQN().eval()
    obs = a_tensores([observacion_desde_vista(v) for _, v in _vistas(6, 8, 3)])
    perm = dict(obs); perm['tableros_otros'] = obs['tableros_otros'].flip(1)
    with torch.no_grad():
        assert torch.allclose(m(obs), m(perm), atol=1e-5)


def test_con_2_jugadores_izquierda_y_derecha_son_el_mismo_rival():
    o = observacion_desde_vista(_vistas(2, 3)[-1][1])
    assert np.array_equal(o['tablero_izq'], o['tablero_der']) and o['tableros_otros'].shape[0] == 0


def test_agente_v5_juega_partidas_completas_con_2_3_y_4_asientos():
    torch.manual_seed(2)
    ag = AgenteDQN(SkyjoDQN())
    for n in (2, 3, 4):
        r = evaluar([ag] + [AgenteHeuristico() if i % 2 else AgenteAleatorio() for i in range(n - 1)],
                    n_partidas=n, semilla=0)
        assert r['_global']['truncadas'] == 0 and r['_global']['jugadores'] == n
        assert all(v['ilegales'] == 0 for k, v in r.items() if k != '_global')
