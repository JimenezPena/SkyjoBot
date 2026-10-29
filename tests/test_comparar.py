import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

import pytest

import comparar_entrenamientos as C

LOG = """Utilizando dispositivo: cuda | jugadores por partida: 4
Barajas simétricas: cada baraja se juega 4 veces con el aprendiz rotando por los asientos
Recompensa final: referencia = mejor de los rivales
== [Tras imitación: 1 DQN vs 3 heurísticos ==
300 partidas, 4 jugadores
agente                   win%  puntos  pos.  cols ilegales
dqn_v5_0                 21.7    27.9  2.67  0.03        0
heuristicos              26.1    24.9  2.42  0.15        0
puntos medios DQN: 27.94 ± 0.83  (heurísticos 24.90)
== [Tras imitación: 1 DQN vs 3 heurísticos ==
300 partidas, 4 jugadores
agente                   win%  puntos  pos.  cols ilegales
dqn_v5_0                 13.5    33.2  2.97  0.03        0
heuristicos              28.8    24.0  2.31  0.16        0
puntos medios DQN: 33.22 ± 0.91  (heurísticos 24.03)
Ep    500/20000 | eps 0.137 | pasos    13250 | pts aprendiz  37.8 | rivales  25.1 | loss 5.5170 | Recompensa aprendiz:    2.9 | 4.8 min
Ep   1000/20000 | eps 0.124 | pasos    26598 | pts aprendiz  37.6 | rivales  25.2 | loss 2.4661 | Recompensa aprendiz:    4.4 | 6.5 min
== [ep 1000] 1 DQN vs 3 heurísticos ==
300 partidas, 4 jugadores
agente                   win%  puntos  pos.  cols ilegales
dqn_v5_0                 12.7    33.2  2.99  0.03        0
heuristicos              29.1    24.0  2.31  0.16        0
puntos medios DQN: 33.17 ± 0.92  (heurísticos 24.02)
== [ep 1000] 1 DQN vs 1 heurísticos ==
300 partidas, 2 jugadores
agente                   win%  puntos  pos.  cols ilegales
dqn_v5_0                 32.3    33.0  1.68  0.06        0
heuristicos              67.7    20.4  1.32  0.20        0
puntos medios DQN: 32.96 ± 1.16  (heurísticos 20.38)
   -> nuevo mejor checkpoint (puntos medios 33.17 con 4 jugadores)
== FINAL 4 jugadores (1 DQN vs 3 heurísticos) ==
300 partidas, 4 jugadores
agente                   win%  puntos  pos.  cols ilegales
dqn_v5_0                 12.7    33.2  2.99  0.03        0
heuristicos              29.1    24.0  2.31  0.16        0
puntos medios DQN: 33.17 ± 0.92  (heurísticos 24.02)
"""


def test_parseo_del_log_de_texto(tmp_path):
    p = tmp_path / 'log.log'
    p.write_text(LOG, encoding='utf-8')
    run = C.leer_run(str(p))
    ev = {(e['episodio'], e['jugadores'], e['fase']): e for e in run['evals']}
    assert set(ev) == {(0, 4, 'tras_bc'), (0, 4, 'tras_calibracion'), (1000, 4, 'entrenamiento'), (1000, 2, 'entrenamiento')}
    e = ev[(1000, 4, 'entrenamiento')]
    assert e['dqn']['puntos'] == 33.17 and e['dqn']['puntos_ee'] == 0.92 and e['heuristicos']['puntos'] == 24.02   # 2 decimales
    assert e['dqn']['posicion'] == 2.99 and abs(e['dqn']['winrate'] - 0.127) < 1e-9 and e['partidas'] == 300
    assert ev[(0, 4, 'tras_bc')]['dqn']['puntos'] == 27.94 and ev[(0, 4, 'tras_calibracion')]['dqn']['puntos'] == 33.22
    assert [r['episodio'] for r in run['train']] == [500, 1000] and run['train'][0]['loss'] == 5.517
    assert run['meta']['recompensa_ref'] == 'mejor' and run['meta']['barajas_simetricas']     # el bloque FINAL no se duplica


def test_jsonl_se_queda_con_la_ultima_aparicion_de_cada_episodio(tmp_path):
    def ev(ep, n, pts, fase='entrenamiento'):
        return dict(tipo='eval', episodio=ep, jugadores=n, fase=fase, partidas=300,
                    dqn=dict(puntos=pts, puntos_ee=0.8, posicion=2.5, winrate=0.2), heuristicos=dict(puntos=24.0))
    filas = [dict(tipo='meta', run='x'), ev(1000, 4, 30.0), dict(tipo='train', episodio=500, pts_aprendiz=30.0, pts_rivales=None),
             ev(1000, 4, 29.0), ev(2000, 4, 28.0)]                  # 1000 repetido (reanudado desde un checkpoint anterior)
    d = tmp_path / 'ck'; d.mkdir()
    (d / 'metricas.jsonl').write_text('\n'.join(json.dumps(f) for f in filas) + '\n', encoding='utf-8')
    run = C.leer_run(str(d))                                        # acepta la carpeta
    assert [(e['episodio'], e['dqn']['puntos']) for e in run['evals']] == [(1000, 29.0), (2000, 28.0)]
    assert run['train'][0]['pts_rivales'] is None


def test_series_y_resumen(tmp_path, capsys):
    p = tmp_path / 'log.log'
    p.write_text(LOG, encoding='utf-8')
    run = C.leer_run(str(p))
    s = C.serie(run, 4, C.EXTRACTORES['puntos'][0])
    assert [x[0] for x in s] == [0, 1000] and s[0][1] == 33.22                      # el BC se omite; la recalibrada es el inicio
    assert [x[0] for x in C.serie(run, 4, C.EXTRACTORES['puntos'][0], inicio=False)] == [1000]
    assert C.serie(run, 4, C.EXTRACTORES['diferencia'][0])[-1][1] == pytest.approx(33.17 - 24.02)
    assert C._media_movil([1.0, 3.0, 5.0], 2) == [1.0, 2.0, 4.0]


def test_script_genera_plots_y_csv(tmp_path):
    pytest.importorskip('matplotlib')
    p = tmp_path / 'log.log'
    p.write_text(LOG, encoding='utf-8')
    out = tmp_path / 'plots'
    C.main(['--run', f'a={p}', '--run', f'b={p}', '--salida', str(out), '--hasta', '1000', '--suavizado', '2'])
    for nombre in ('puntos', 'posicion', 'winrate', 'diferencia', 'entrenamiento'):
        assert (out / f'{nombre}.png').stat().st_size > 1000
    assert (out / 'evaluaciones.csv').read_text(encoding='utf-8').count('\n') == 1 + 2 * 4      # cabecera + 2 ejecuciones x 4 evals
