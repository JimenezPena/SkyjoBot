"""v6.0: red de grafos sobre las 12 cartas. Misma observación/entrenamiento que v5.0; aquí se prueba la red."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

from skyjo.vista import construir_vista
from skyjo.v6_0 import (SkyjoDQN, SkyjoEnv, N_ESCALARES, N_ACCIONES, observacion_desde_vista, a_tensores)

FILAS, COLS = [1, 2, 0], [2, 0, 3, 1]                 # permutaciones de filas y columnas del tablero 3x4


def _obs_reales(n, pasos=6, semilla=0):
    rng = np.random.default_rng(semilla)
    env = SkyjoEnv(num_jugadores=n); env.reset()
    out = []
    for _ in range(pasos):
        out.append(observacion_desde_vista(construir_vista(env)))
        env.step(int(rng.choice(np.flatnonzero(env.get_action_mask(env.turno_actual)))))
    return out


def _obs_aleatoria(B, k, semilla=0):
    """Tableros arbitrarios (valores en [-1, 1], estados -1/0/1): comprueba la matemática de la red, no el juego."""
    g = torch.Generator().manual_seed(semilla)

    def tab(*lead):
        val = torch.rand(*lead, 3, 4, generator=g) * 2 - 1
        est = torch.randint(-1, 2, (*lead, 3, 4), generator=g).float()
        return torch.stack([val, est], dim=len(lead))

    return {'tablero_yo': tab(B), 'tablero_izq': tab(B), 'tablero_der': tab(B), 'tableros_otros': tab(B, k),
            'escalares': torch.randn(B, N_ESCALARES, generator=g)}


def _permutar(t, filas, cols):
    """Reordena las casillas de tableros [..., 3, 4]: nuevo[f, c] = viejo[filas[f], cols[c]]."""
    return t[..., filas, :][..., cols]


def test_formas_para_2_a_8_jugadores_y_parametros_independientes_de_n():
    torch.manual_seed(0)
    m = SkyjoDQN()
    for n in range(2, 9):
        obs = _obs_reales(n)
        assert obs[0]['tableros_otros'].shape == (max(n - 3, 0), 2, 3, 4)
        with torch.no_grad():
            q = m(a_tensores(obs))
        assert q.shape == (len(obs), N_ACCIONES) and torch.isfinite(q).all()
    assert SkyjoDQN(2).state_dict().keys() == SkyjoDQN(8).state_dict().keys()


def test_la_red_es_equivariante_a_permutar_filas_y_columnas_de_mi_tablero():
    torch.manual_seed(1)
    m = SkyjoDQN().eval()
    obs = _obs_aleatoria(5, 2)
    perm = dict(obs, tablero_yo=_permutar(obs['tablero_yo'], FILAS, COLS))
    with torch.no_grad():
        q, qp = m(obs), m(perm)
    idx = torch.tensor([FILAS[f] * 4 + COLS[c] for f in range(3) for c in range(4)])    # casilla vieja de cada nueva
    assert torch.allclose(qp[:, :2], q[:, :2], atol=1e-5)                               # robar: no cambia
    assert torch.allclose(qp[:, 2:14], q[:, 2:14][:, idx], atol=1e-5)                   # colocar
    assert torch.allclose(qp[:, 14:], q[:, 14:][:, idx], atol=1e-5)                     # descartar y destapar


def test_la_red_es_invariante_a_permutar_las_casillas_de_los_rivales_y_el_orden_de_los_otros():
    torch.manual_seed(2)
    m = SkyjoDQN().eval()
    obs = _obs_aleatoria(4, 3)
    perm = dict(obs, tablero_izq=_permutar(obs['tablero_izq'], FILAS, COLS),
                tablero_der=_permutar(obs['tablero_der'], [2, 1, 0], [3, 2, 1, 0]),
                tableros_otros=_permutar(obs['tableros_otros'], [1, 0, 2], [1, 2, 3, 0])[:, [2, 0, 1]])
    with torch.no_grad():
        assert torch.allclose(m(obs), m(perm), atol=1e-5)


def test_las_aristas_son_de_columna_cambiar_otra_columna_no_altera_una_casilla_sin_contexto():
    """Sin contexto global, el embedding de una carta solo depende de su columna: cambiar la columna 3 no mueve la 0."""
    torch.manual_seed(3)
    m = SkyjoDQN().eval()
    x = _obs_aleatoria(2, 0)['tablero_yo']
    y = x.clone(); y[:, :, :, 3] = torch.randn_like(y[:, :, :, 3]).clamp(-1, 1)         # otra columna cualquiera
    with torch.no_grad():
        hx, hy = m._grafo(x), m._grafo(y)
    assert torch.allclose(hx[:, :, :3], hy[:, :, :3], atol=1e-6)
    assert not torch.allclose(hx[:, :, 3], hy[:, :, 3], atol=1e-3)


def test_la_red_distingue_una_pareja_en_columna_de_dos_cartas_sueltas():
    """Para poder aprender los tríos hay que ver si las cartas de la columna coinciden: mismos valores repartidos en
    columna o en fila dan embeddings distintos."""
    torch.manual_seed(4)
    m = SkyjoDQN().eval()
    x = torch.zeros(1, 2, 3, 4); x[:, 1] = 1.0
    a, b = x.clone(), x.clone()
    a[0, 0, :, 0] = torch.tensor([0.5, 0.5, -0.3])          # dos iguales en la misma columna
    b[0, 0, :, 0] = torch.tensor([0.5, -0.3, 0.0]); b[0, 0, 0, 1] = 0.5   # el 0.5 repetido, pero en otra columna
    with torch.no_grad():
        ra, rb = SkyjoDQN._resumen(m._grafo(a)), SkyjoDQN._resumen(m._grafo(b))
    assert not torch.allclose(ra, rb, atol=1e-4)


def test_numero_de_parametros_es_comparable_a_v5():
    from skyjo.v5_0 import SkyjoDQN as V5
    n6, n5 = (sum(p.numel() for p in M().parameters()) for M in (SkyjoDQN, V5))
    assert 0.3 * n5 < n6 < 1.5 * n5, (n6, n5)


def _config_mini(tmp_path, jugadores):
    from skyjo.v6_0 import CONFIG
    return dict(CONFIG, JUGADORES=jugadores, NUM_EPISODIOS=4, BUFFER_CAPACITY=2000, BATCH_SIZE=32, WARMUP_STEPS=40,
                TRAIN_FREQ=4, TARGET_UPDATE_FREQ=50, LOG_EVERY=2, EVAL_EVERY=2, EVAL_GAMES=2,
                EVAL_JUGADORES=[jugadores], CKPT_DIR=str(tmp_path / 'ck'), SAVE_PATH=str(tmp_path / 'm.pth'),
                SAVE_EVERY=2, IMIT_EPISODIOS=3, IMIT_EPOCHS=1, CALIBRATION_PRESTEPS=3, DEMO_STEPS=50, RIVALES=None)


def test_entrenamiento_mini_corre_guarda_y_el_modelo_es_usable_con_4_y_2_jugadores(tmp_path):
    import skyjo.v6_0 as v6
    for n in (4, 2):
        d = tmp_path / str(n); d.mkdir()
        cfg = _config_mini(d, n)
        v6.entrenar_skyjo(cfg)
        assert os.path.exists(cfg['SAVE_PATH']) and os.path.exists(cfg['CKPT_DIR'] + '/best.pth')
        net = v6.SkyjoDQN(); net.load_state_dict(torch.load(cfg['SAVE_PATH']))
        r = v6.evaluar_vs_heuristicos(net, "cpu", n, n_partidas=n)
        assert r['_global']['truncadas'] == 0 and r['_dqn']['ilegales'] == 0
