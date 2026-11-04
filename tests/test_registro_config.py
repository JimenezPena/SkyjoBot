import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from skyjo.registro_config import diferencias, registrar

DEFECTO = {'RECOMPENSA_REF': 'media', 'RECOMPENSA_FINAL': 1.5, 'NUM_EPISODIOS': 20000, 'LR': 1e-4}


def _lineas(ruta):
    return [json.loads(l) for l in open(ruta, encoding='utf-8')]


def test_registra_overrides_y_reanudacion_con_cambios(tmp_path, capsys):
    d = str(tmp_path / 'ck')
    cfg = dict(DEFECTO, RECOMPENSA_REF='mejor', NUM_EPISODIOS=30000, LR=np.float64(1e-4))
    ruta, cambios = registrar(d, 'v5_0', cfg, DEFECTO, argv=['--set', 'RECOMPENSA_REF=mejor'], commit='abc')
    assert cambios == {} and 'AVISO' not in capsys.readouterr().out
    r = _lineas(ruta)[0]
    assert r['overrides'] == {'RECOMPENSA_REF': ['media', 'mejor'], 'NUM_EPISODIOS': [20000, 30000]}
    assert r['config']['RECOMPENSA_REF'] == 'mejor' and r['reanudacion'] is False and r['commit'] == 'abc'

    # reanudación olvidando el --set de recompensa y alargando NUM_EPISODIOS (esto último es esperable)
    cfg2 = dict(DEFECTO, NUM_EPISODIOS=40000, RECOMPENSA_FINAL=5.0)
    ruta, cambios = registrar(d, 'v5_0', cfg2, DEFECTO, argv=[])
    out = capsys.readouterr().out
    assert set(cambios) == {'RECOMPENSA_REF', 'RECOMPENSA_FINAL', 'NUM_EPISODIOS'}
    assert 'RECOMPENSA_FINAL: 1.5 -> 5.0' in out and 'NUM_EPISODIOS' not in out
    lineas = _lineas(ruta)
    assert len(lineas) == 2 and lineas[1]['reanudacion'] is True


def test_diferencias_ignora_el_tipo_numpy():
    assert diferencias({'a': 1, 'b': 2}, {'a': np.int64(1), 'c': 3}) == {'b': (2, None), 'c': (None, 3)}
