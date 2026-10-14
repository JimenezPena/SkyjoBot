"""Observación v5.0: independiente del nº de jugadores y construida SOLO desde la vista pública.

Tableros, en orden relativo al asiento que decide (ver skyjo.vista):
  yo (0) | izq (1: el siguiente, recibe mis descartes) | der (n-1: el anterior, de quien viene el descarte) | otros (2..n-2)
Con 2 jugadores izq y der son el mismo rival; con 3 no hay 'otros' (k = max(n-3, 0)).
Cada tablero = 2 planos 3x4: valor normalizado (solo cartas visibles) y estado (1 visible, 0 oculta, -1 eliminada).

Escalares (N_ESCALARES = 44):
  globales (27): carta en mano, tope del descarte, fase, origen, conteo de lo no visto (15), cartas en el mazo,
                 hay cerrador, quién cerró (yo/izq/der/otro), turnos restantes, nº de jugadores
  por asiento yo/izq/der (3x4): ocultas, puntos visibles, puntos estimados, diferencia estimada con la mía
  resumen de 'otros' (5): cuántos, ocultas medias y mínimas, diferencia mínima y media de puntos estimados
"""
import numpy as np

from .env import TOTALES_INICIALES, VALORES_CARTA

N_ESCALARES = 44
MAX_JUGADORES = 8


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
    return 5.0 if total <= 0 else float((conteo * VALORES_CARTA).sum() / total)


def _tableros(vista):
    val = np.where(vista.estado == 1, (vista.valores - 5.0) / 7.0, 0.0).astype(np.float32)
    est = vista.estado.astype(np.float32)
    return np.stack([val, est], axis=1)                    # (n, 2, 3, 4)


def observacion_desde_vista(vista):
    n = vista.n
    tab = _tableros(vista)
    izq, der = 1, n - 1
    otros = tab[2:n - 1]                                   # (max(n-3,0), 2, 3, 4)

    c_mano = ((float(vista.mano) - 5.0) / 7.0) if vista.mano is not None else 0.0
    top = vista.top_descarte
    top_d = ((float(top) - 5.0) / 7.0) if top is not None else 0.0
    conteo = _conteo(vista)
    e_oculta = _valor_esperado(conteo)

    ocultas = vista.ocultas.astype(np.float32)
    vis_pts = vista.puntos_visibles.astype(np.float32)
    est = vis_pts + ocultas * e_oculta
    dif = np.clip((est - est[0]) / 30.0, -2.0, 2.0)

    cer = vista.cerrador
    globales = np.concatenate([
        np.array([c_mano, top_d, vista.fase / 2.0, 1.0 if vista.origen == 'DESCARTE' else 0.0], dtype=np.float32),
        conteo / TOTALES_INICIALES,
        np.array([vista.n_mazo / 150.0,
                  1.0 if cer is not None else 0.0,
                  1.0 if cer == 0 else 0.0,
                  1.0 if cer == izq else 0.0,
                  1.0 if (cer == der and cer != izq) else 0.0,       # con 2 jugadores es 'izq'
                  1.0 if (cer is not None and 2 <= cer <= n - 2) else 0.0,
                  max(vista.turnos_restantes, 0) / max(n - 1, 1),
                  n / MAX_JUGADORES], dtype=np.float32)])

    def asiento(i):
        return np.array([ocultas[i] / 12.0, vis_pts[i] / 50.0, est[i] / 50.0, dif[i]], dtype=np.float32)

    k = max(n - 3, 0)
    if k:
        o = np.arange(2, n - 1)
        resumen = np.array([k / MAX_JUGADORES, ocultas[o].mean() / 12.0, ocultas[o].min() / 12.0,
                            dif[o].min(), dif[o].mean()], dtype=np.float32)
    else:
        resumen = np.zeros(5, dtype=np.float32)

    escalares = np.concatenate([globales, asiento(0), asiento(izq), asiento(der), resumen]).astype(np.float32)
    assert escalares.shape[0] == N_ESCALARES, escalares.shape
    return {'tablero_yo': tab[0], 'tablero_izq': tab[izq], 'tablero_der': tab[der],
            'tableros_otros': otros, 'escalares': escalares}


def a_tensores(observaciones, device="cpu"):
    """Lista de observaciones (todas con el mismo nº de jugadores) -> dict de tensores con dimensión de batch."""
    import torch
    return {k: torch.from_numpy(np.stack([o[k] for o in observaciones])).to(device) for k in observaciones[0]}
