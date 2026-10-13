"""Observación del modelo 4.x construida únicamente a partir de la vista pública (skyjo.vista.Vista).
Es exactamente la misma observación que calculaba el entorno (28 escalares), pero sin acceder a él."""
import numpy as np

from .env import VALORES_CARTA, TOTALES_INICIALES, N_ESCALARES


def _conteo(vista):
    """Nº de cartas de cada valor que AÚN NO se han visto (mazo + boca abajo)."""
    vistas = np.zeros(15, dtype=np.float32)
    if len(vista.descartes):
        vistas += np.bincount(vista.descartes.astype(np.int64) + 2, minlength=15)[:15]
    vis = vista.valores[vista.estado == 1]
    if vis.size:
        vistas += np.bincount(vis.astype(np.int64) + 2, minlength=15)[:15]
    if vista.mano is not None:
        vistas[vista.mano + 2] += 1.0
    return np.maximum(TOTALES_INICIALES - vistas, 0.0)


def _valor_esperado(conteo):
    total = float(conteo.sum())
    if total <= 0:
        return 5.0
    return float((conteo * VALORES_CARTA).sum() / total)


def observacion_desde_vista(vista):
    n = vista.n
    valores_tableros = np.where(vista.estado == 1, (vista.valores - 5.0) / 7.0, 0.0).astype(np.float32)
    visibilidad_tableros = vista.estado.astype(np.float32)

    c_mano = ((float(vista.mano) - 5.0) / 7.0) if vista.mano is not None else 0.0
    top = vista.top_descarte
    top_descarte = ((float(top) - 5.0) / 7.0) if top is not None else 0.0
    fase = float(vista.fase) / 2.0
    origen_flag = 1.0 if vista.origen == 'DESCARTE' else 0.0
    basicos = np.array([c_mano, top_descarte, fase, origen_flag], dtype=np.float32)

    conteo = _conteo(vista)
    vector_conteo = conteo / TOTALES_INICIALES
    e_oculta = _valor_esperado(conteo)

    ocultas = vista.ocultas                                   # (n,) en orden relativo (0 = yo)
    vis_pts = vista.puntos_visibles
    est = vis_pts + ocultas * e_oculta                        # puntos estimados por jugador
    ocultas_riv, est_riv = ocultas[1:], est[1:]

    extra = np.array([
        1.0 if vista.cerrador is not None else 0.0,
        max(vista.turnos_restantes, 0) / 3.0,
        ocultas[0] / 12.0,
        float(np.mean(ocultas_riv)) / 12.0,
        min(ocultas_riv) / 12.0,
        float(vis_pts[0]) / 50.0,
        est[0] / 50.0,
        float(np.clip((min(est_riv) - est[0]) / 30.0, -2.0, 2.0)),
        vista.n_mazo / 150.0,
    ], dtype=np.float32)

    escalares = np.concatenate([basicos, vector_conteo, extra]).astype(np.float32)
    assert escalares.shape[0] == N_ESCALARES
    return {
        'valores_tableros': valores_tableros,
        'visibilidad_tableros': visibilidad_tableros,
        'escalares': escalares,
    }
