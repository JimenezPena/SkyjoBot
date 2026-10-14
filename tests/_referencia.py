"""Implementaciones ORIGINALES (antes de la vista), solo para comprobar que no ha cambiado nada."""
import random

import numpy as np

from skyjo.v4_2.env import TOTALES_INICIALES, N_ESCALARES


def obs_original(env, jugador_id=None):
    """SkyjoEnv.obtener_observacion de la 4.1/4.2 (leía el entorno directamente)."""
    if jugador_id is None:
        jugador_id = env.turno_actual
    num_j = env.num_jugadores
    indices_ordenados = [(jugador_id + i) % num_j for i in range(num_j)]
    tableros_rotados = env.tableros[indices_ordenados]
    visible_mask = (tableros_rotados[:, :, :, 1] == 1)
    valores_tableros = np.where(visible_mask, (tableros_rotados[:, :, :, 0] - 5.0) / 7.0, 0.0).astype(np.float32)
    visibilidad_tableros = tableros_rotados[:, :, :, 1].astype(np.float32)
    c_mano = ((float(env.carta_en_mano) - 5.0) / 7.0) if env.carta_en_mano is not None else 0.0
    top_descarte = ((float(env.descartes[-1]) - 5.0) / 7.0) if len(env.descartes) > 0 else 0.0
    fase = float(env.fase_turno) / 2.0
    origen_flag = 1.0 if env.origen_robo == 'DESCARTE' else 0.0
    escalares_basicos = np.array([c_mano, top_descarte, fase, origen_flag], dtype=np.float32)
    conteo = env._conteo_restantes()
    vector_conteo = conteo / TOTALES_INICIALES
    e_oculta = env._valor_esperado_oculta(conteo)
    rivales = [j for j in range(num_j) if j != jugador_id]
    ocultas = {j: int(np.sum(env.tableros[j, :, :, 1] == 0)) for j in range(num_j)}
    ocultas_rivales = [ocultas[j] for j in rivales]
    est_propio = env._puntos_estimados(jugador_id, e_oculta)
    est_rivales = [env._puntos_estimados(j, e_oculta) for j in rivales]
    extra = np.array([
        1.0 if env.cerrador_id is not None else 0.0,
        max(env.turnos_restantes, 0) / 3.0,
        ocultas[jugador_id] / 12.0,
        float(np.mean(ocultas_rivales)) / 12.0,
        min(ocultas_rivales) / 12.0,
        env._calcular_puntos_tablero(jugador_id) / 50.0,
        est_propio / 50.0,
        float(np.clip((min(est_rivales) - est_propio) / 30.0, -2.0, 2.0)),
        len(env.mazo) / 150.0,
    ], dtype=np.float32)
    escalares = np.concatenate([escalares_basicos, vector_conteo, extra]).astype(np.float32)
    assert escalares.shape[0] == N_ESCALARES
    return {'valores_tableros': valores_tableros, 'visibilidad_tableros': visibilidad_tableros, 'escalares': escalares}


def heuristico_original(env, mask):
    """El heurístico de antes: leía el entorno."""
    j = env.turno_actual
    e = env._valor_esperado_oculta()

    def analisis(carta):
        t = env.tableros[j]
        gan = np.zeros((3, 4), np.float32)
        val = np.zeros((3, 4), bool)
        for c in range(4):
            vis, vals = t[:, c, 1], t[:, c, 0]
            if vis[0] == -1:
                continue
            for r in range(3):
                val[r, c] = True
                old = float(vals[r]) if vis[r] == 1 else e
                otros = [k for k in range(3) if k != r]
                if all(vis[k] == 1 and vals[k] == carta for k in otros):
                    gan[r, c] = 2 * carta + old
                else:
                    gan[r, c] = old - carta
        return gan, val

    if env.fase_turno == 1:
        gan, val = analisis(int(env.descartes[-1]))
        return 0 if (val.any() and gan[val].max() >= 3) else 1
    gan, val = analisis(int(env.carta_en_mano))
    gan = np.where(val, gan, -1e9)
    mejor = int(np.argmax(gan))
    ocultas = np.flatnonzero(mask[14:])
    if env.origen_robo == 'MAZO' and gan.flat[mejor] < 2 and len(ocultas) > 0:
        return 14 + int(random.choice(ocultas))
    return 2 + mejor
