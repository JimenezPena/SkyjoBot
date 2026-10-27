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


def _config_mini(tmp_path, jugadores):
    from skyjo.v5_0 import CONFIG
    c = dict(CONFIG, JUGADORES=jugadores, NUM_EPISODIOS=4, BUFFER_CAPACITY=2000, BATCH_SIZE=32, WARMUP_STEPS=40,
             TRAIN_FREQ=4, TARGET_UPDATE_FREQ=50, LOG_EVERY=2, EVAL_EVERY=2, EVAL_GAMES=2,
             EVAL_JUGADORES=[jugadores], CKPT_DIR=str(tmp_path / 'ck'), SAVE_PATH=str(tmp_path / 'm.pth'),
             SAVE_EVERY=2, IMIT_EPISODIOS=3, IMIT_EPOCHS=1, CALIBRATION_PRESTEPS=3, DEMO_STEPS=50, RIVALES=None)
    return c


def test_entrenamiento_self_play_con_4_y_2_jugadores_corre_y_guarda_un_modelo_usable(tmp_path):
    import skyjo.v5_0 as v5
    for n in (4, 2):
        d = tmp_path / str(n); d.mkdir()
        cfg = _config_mini(d, n)
        v5.entrenar_skyjo(cfg)
        assert os.path.exists(cfg['SAVE_PATH']) and os.path.exists(cfg['CKPT_DIR'] + '/best.pth')
        net = v5.SkyjoDQN(); net.load_state_dict(torch.load(cfg['SAVE_PATH']))
        r = v5.evaluar_vs_heuristicos(net, "cpu", n, n_partidas=n)
        assert r['_global']['truncadas'] == 0 and r['_dqn']['ilegales'] == 0


def test_evaluar_rota_el_dqn_por_todos_los_asientos_de_cada_baraja(monkeypatch):
    import skyjo.v5_0.agents as ag
    from skyjo.heuristico import AgenteHeuristico

    class Clon(AgenteHeuristico):                       # un 'DQN' que juega como el heurístico
        nombre = 'clon'

        def __init__(self, net, device='cpu'):
            super().__init__()

    monkeypatch.setattr(ag, 'AgenteDQN', Clon)
    for n in (2, 4):
        r = ag.evaluar_vs_heuristicos(None, 'cpu', n, n_partidas=3 * n, semilla=7)
        assert r['_global']['partidas'] == 3 * n and r['_global']['truncadas'] == 0
        # con el DQN en todos los asientos de cada baraja, un clon del heurístico da los mismos puntos que los rivales
        assert abs(r['clon']['puntos'] - r['heuristicos']['puntos']) < 1e-9
        assert r == ag.evaluar_vs_heuristicos(None, 'cpu', n, n_partidas=3 * n, semilla=7)


def test_el_buffer_guarda_y_devuelve_observaciones_de_cualquier_n():
    from skyjo.v5_0 import ReplayBuffer
    for n in (2, 3, 5):
        v = _vistas(n, 4)
        obs = [observacion_desde_vista(x) for _, x in v]
        b = ReplayBuffer(10, obs[0])
        for o in obs:
            b.push(o, 1, 0.5, o, False, v[0][1].mask)
        o, a, r, no, d, m = b.sample(3)
        assert o['tableros_otros'].shape == (3, max(n - 3, 0), 2, 3, 4) and a.shape == (3, 1) and m.shape == (3, N_ACCIONES)
        assert SkyjoDQN()(o).shape == (3, N_ACCIONES)


def test_pesos_de_rivales_se_interpolan_y_sin_checkpoints_se_usa_el_heuristico():
    from skyjo.v5_0.rivales import pesos_en, PoolRivales
    from skyjo.heuristico import AgenteHeuristico
    cfg = {'RIVALES': {'heuristico': 1.0}, 'RIVALES_FIN': {'heuristico': 0.0, 'actual': 1.0}}
    assert pesos_en(cfg, 0.0)['heuristico'] == 1.0
    p = pesos_en(cfg, 0.5)
    assert abs(p['heuristico'] - 0.5) < 1e-9 and abs(p['actual'] - 0.5) < 1e-9 and p['checkpoint'] == 0.0
    pool = PoolRivales(SkyjoDQN(), "cpu")
    assert isinstance(pool.elegir({'heuristico': 0, 'actual': 0, 'checkpoint': 1}), AgenteHeuristico)   # pool vacío
    assert isinstance(pool.elegir({'heuristico': 0, 'actual': 0, 'checkpoint': 0}), AgenteHeuristico)   # sin pesos


def test_entrenamiento_contra_mezcla_de_rivales_con_asiento_aleatorio(tmp_path):
    import skyjo.v5_0 as v5
    torch.manual_seed(0)
    torch.save(v5.SkyjoDQN().state_dict(), tmp_path / "externo.pth")
    for n, k in ((4, 2), (2, 1)):
        d = tmp_path / f"{n}_{k}"; d.mkdir()
        cfg = dict(_config_mini(d, n), RIVALES={'heuristico': 0.4, 'actual': 0.3, 'checkpoint': 0.3},
                   RIVALES_FIN={'heuristico': 0.2, 'actual': 0.4, 'checkpoint': 0.4},
                   RIVALES_CKPT=[str(tmp_path / "externo.pth")], ASIENTOS_APRENDIZ=k, POOL_EVERY=2, POOL_MAX=2)
        v5.entrenar_skyjo(cfg)
        assert os.path.exists(cfg['SAVE_PATH']) and os.path.exists(cfg['CKPT_DIR'] + '/pool.pt')
        assert len(torch.load(cfg['CKPT_DIR'] + '/pool.pt')) <= 2
        v5.entrenar_skyjo(dict(cfg, NUM_EPISODIOS=6))        # reanuda desde el checkpoint (con el pool guardado)


def test_recompensa_de_victoria_es_opcional_y_suma_cero_extra():
    from skyjo.v5_0.imitacion import recompensas_finales
    base = {'RECOMPENSA_FINAL': 1.5}
    p = [10, 20, 30, 10]
    r0 = recompensas_finales(p, base)
    r1 = recompensas_finales(p, dict(base, RECOMPENSA_VICTORIA=20))
    d = r1 - r0
    assert abs(d.sum()) < 1e-9 and d[0] == d[3] > 0 > d[1] == d[2]
