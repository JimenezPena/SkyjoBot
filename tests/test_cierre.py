"""Estadísticas de cierre de ronda: cartas boca abajo de cada asiento al cerrar (-1 el cerrador) y turnos hasta cerrar."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest

from skyjo.heuristico import AgenteHeuristico
from skyjo.partida import Resultado, jugar_partida


def _partida(n, semilla):
    decisiones = [0] * n

    def contar(asiento, agente, vista, accion):
        decisiones[asiento] += 1

    return jugar_partida([AgenteHeuristico() for _ in range(n)], semilla=semilla, on_paso=contar), decisiones


@pytest.mark.parametrize('n', [2, 3, 4])
def test_ocultas_y_turnos_al_cierre(n):
    for semilla in range(6):
        r, decisiones = _partida(n, semilla)
        assert not r.truncada and r.cerrador is not None
        assert len(r.ocultas_cierre) == n
        assert r.ocultas_cierre[r.cerrador] == -1                              # el cerrador se marca con -1
        assert all(0 <= o <= 12 for k, o in enumerate(r.ocultas_cierre) if k != r.cerrador)
        assert r.turnos_cierre == decisiones[r.cerrador] // 2 >= 1             # tras cerrar, el cerrador ya no juega
        # vuelta final: los asientos anteriores al cerrador juegan un turno más que él; los posteriores, los mismos
        assert all(d // 2 == r.turnos_cierre + (k < r.cerrador) for k, d in enumerate(decisiones) if k != r.cerrador)


def test_partida_truncada_no_tiene_estadisticas_de_cierre():
    r = jugar_partida([AgenteHeuristico(), AgenteHeuristico()], max_pasos=5, semilla=0)
    assert r.truncada and r.ocultas_cierre is None and r.turnos_cierre is None


def test_resumen_separa_cierre_ocultas_y_turnos():
    pytest.importorskip('torch')
    from skyjo.v6_0.agents import _acum, _resumen, _sumar, _linea_cierre

    def res(cerrador, ocultas, turnos):
        return Resultado(puntos=[10, 20, 30], posiciones=[1, 2, 3], ganadores=[0], pasos=40, truncada=False, ilegales=[0, 0, 0],
                         columnas=[0, 0, 0], cerrador=cerrador, ocultas_cierre=ocultas, turnos_cierre=turnos)

    a = _acum()
    for r in (res(0, [-1, 2, 4], 8), res(1, [3, -1, 1], 10), res(1, [1, -1, 0], 6),
              Resultado(puntos=[0, 0, 0], posiciones=[1, 1, 1], ganadores=[0, 1, 2], pasos=3000, truncada=True,
                        ilegales=[0, 0, 0], columnas=[0, 0, 0], cerrador=None)):     # truncada: no cuenta
        _sumar(a, r, 0)
    s = _resumen(a)
    assert s['partidas'] == 4 and s['cierra'] == pytest.approx(1 / 3)                 # cierra 1 de las 3 rondas cerradas
    assert s['turnos_cierra'] == 8
    assert s['ocultas_cierre'] == pytest.approx((3 + 1) / 2)                          # solo cuando cierra otro (sin los -1)
    vacio = _resumen(_acum())
    assert vacio['cierra'] is None and vacio['ocultas_cierre'] is None and vacio['turnos_cierra'] is None
    assert 'cierra 33.3%' in _linea_cierre(s, vacio) and 'n/d' in _linea_cierre(s, vacio)
