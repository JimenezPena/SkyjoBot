"""Registro de métricas del entrenamiento: <CKPT_DIR>/metricas.jsonl, una línea JSON por registro.

Sirve para comparar ejecuciones sin parsear logs: python comparar_entrenamientos.py --run "nombre=<CKPT_DIR>" ...
Se abre en modo añadir: si el entrenamiento se reanuda desde un checkpoint, los registros nuevos se suman a los
anteriores (el lector se queda con la última aparición de cada episodio). Si falla la escritura (disco lleno, Drive
desconectado...) se avisa por pantalla y el entrenamiento sigue.

Tipos de registro:
  meta        run, fecha, config (CONFIG completo), jugadores_eval, reanudado, inicio_episodio
  imitacion   acc (exactitud del behaviour cloning)
  eval        episodio, jugadores, fase ('tras_bc' | 'tras_calibracion' | 'entrenamiento'), partidas, truncadas,
              dqn{partidas, winrate, puntos, puntos_ee, posicion, columnas, ilegales}, heuristicos{...} (media de los rivales)
              Con BARAJAS_SIMETRICAS las evaluaciones juegan cada baraja con el DQN en todos los asientos (v5_0/agents.py).
  train       episodio, eps, pasos, minutos, episodios (tamaño de la ventana), pts_aprendiz, pts_rivales (null en
              self-play), loss (media de los últimos 500 pasos de optimización), rec_shaping (recompensa intermedia
              acumulada por partida, media de la ventana) y rec_final (recompensa final de los aprendices, media de la
              ventana). La recompensa depende de RECOMPENSA_REF / RECOMPENSA_FINAL / RECOMPENSA_VICTORIA: no es
              comparable entre ejecuciones que las cambien; los puntos sí.
"""
import json
import math
import numbers
import os
import time


def _json(o):
    try:
        return float(o)
    except (TypeError, ValueError):
        return str(o)


def _limpio(x):
    """NaN/inf -> None (no son JSON estándar) y numpy -> tipos de Python, de forma recursiva."""
    if isinstance(x, dict):
        return {k: _limpio(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_limpio(v) for v in x]
    if isinstance(x, bool) or x is None or isinstance(x, str):
        return x
    if isinstance(x, numbers.Integral):                  # int y enteros de numpy
        return int(x)
    if isinstance(x, numbers.Real):                      # float y flotantes de numpy
        f = float(x)
        return None if (math.isnan(f) or math.isinf(f)) else f
    return str(x)


class RegistroMetricas:
    def __init__(self, ruta):
        self.ruta = ruta
        try:
            os.makedirs(os.path.dirname(os.path.abspath(ruta)), exist_ok=True)
        except OSError as e:
            print(f"[métricas] no se pudo crear la carpeta de {ruta}: {e}")

    def _escribir(self, tipo, **datos):
        try:
            with open(self.ruta, 'a', encoding='utf-8') as f:
                f.write(json.dumps(_limpio(dict(tipo=tipo, **datos)), ensure_ascii=False, default=_json) + '\n')
        except OSError as e:
            print(f"[métricas] no se pudo escribir en {self.ruta}: {e}")

    def meta(self, config, run=None, **extra):
        run = run or os.path.splitext(os.path.basename(str(config.get('SAVE_PATH', 'run'))))[0]
        self._escribir('meta', run=run, fecha=time.strftime('%Y-%m-%d %H:%M:%S'), config=config, **extra)

    def imitacion(self, acc):
        self._escribir('imitacion', acc=acc)

    def evaluacion(self, episodio, jugadores, res, fase='entrenamiento'):
        """res = salida de evaluar_vs_heuristicos."""
        g = res.get('_global', {})
        self._escribir('eval', episodio=episodio, jugadores=jugadores, fase=fase, partidas=g.get('partidas'),
                       truncadas=g.get('truncadas'), dqn=res['_dqn'], heuristicos=res.get('heuristicos', {}))

    def entrenamiento(self, episodio, **datos):
        self._escribir('train', episodio=episodio, **datos)
