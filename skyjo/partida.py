"""Árbitro de partidas con N asientos (2-8) y agentes de cualquier tipo (humano, modelo, heurístico...).

  from skyjo.partida import jugar_partida, evaluar
  r = jugar_partida([AgenteHeuristico(), AgenteAleatorio(), AgenteHeuristico(), AgenteAleatorio()])
  res = evaluar([agente_a, agente_b, agente_c, agente_d], n_partidas=240)    # rota los asientos

Cada agente solo recibe la Vista pública. Una acción ilegal se sustituye por una al azar y se cuenta.
"""
import itertools
import random
from dataclasses import dataclass, field
from typing import List, Optional

from .vista import construir_vista


def _motor():
    from skyjo.v4_2 import SkyjoEnv           # las reglas son iguales en todas las versiones
    return SkyjoEnv


@dataclass
class Resultado:
    puntos: List[int]                 # por asiento (menos es mejor), con la penalización al cerrador
    posiciones: List[int]             # 1 = mejor; los empates comparten posición
    ganadores: List[int]
    pasos: int
    truncada: bool                    # superó max_pasos sin terminar
    ilegales: List[int]
    columnas: List[int]               # columnas eliminadas por asiento
    cerrador: Optional[int]
    extra: dict = field(default_factory=dict)
    penalizada: bool = False          # el cerrador no tenía la puntuación más baja y se le dobló


def jugar_partida(agentes, motor=None, max_pasos=3000, on_paso=None):
    """on_paso(asiento, agente, vista, accion) se llama antes de cada jugada (para mostrarla, registrar...)."""
    import numpy as np
    n = len(agentes)
    if not 2 <= n <= 8:
        raise ValueError("Skyjo se juega con 2 a 8 jugadores")
    env = (motor or _motor())(num_jugadores=n)
    env.reset()
    ilegales, pasos = [0] * n, 0
    while not env.game_over and pasos < max_pasos:
        j = env.turno_actual
        vista = construir_vista(env, j)
        a = int(agentes[j](vista))
        if not (0 <= a < len(vista.mask)) or not vista.mask[a]:
            ilegales[j] += 1
            a = int(random.choice(vista.acciones_legales()))
        if on_paso:
            on_paso(j, agentes[j], vista, a)
        env.step(a)
        pasos += 1
    truncada = not env.game_over
    columnas = [int((env.tableros[k, :, :, 1] == -1).sum()) // 3 for k in range(n)]   # antes de destapar todo
    puntos = [int(p) for p in env.calcular_puntuaciones_finales()]
    mejor = min(puntos)
    c = env.cerrador_id
    penalizada = bool(c is not None and puntos[c] > 0 and 2 * min(q for k, q in enumerate(puntos) if k != c) <= puntos[c])
    extra = {'regalos': list(env.regalos)} if hasattr(env, 'regalos') else {}
    return Resultado(puntos=puntos, posiciones=[1 + sum(q < p for q in puntos) for p in puntos],
                     ganadores=[k for k, p in enumerate(puntos) if p == mejor], pasos=pasos, truncada=truncada,
                     ilegales=ilegales, columnas=columnas, cerrador=env.cerrador_id, extra=extra, penalizada=penalizada)


def _nombres(agentes):
    base = [getattr(a, 'nombre', type(a).__name__) for a in agentes]
    vistos, out = {}, []
    for b in base:
        vistos[b] = vistos.get(b, 0) + 1
        out.append(b if base.count(b) == 1 else f"{b}#{vistos[b]}")
    return out


def evaluar(agentes, n_partidas=100, motor=None, max_pasos=3000, semilla=None, detalle=False):
    """Juega n_partidas rotando los asientos (todas las permutaciones si n <= 4, para que cada agente tenga
    todos los vecinos posibles) y devuelve métricas por agente. Los empates reparten la victoria.
    Con detalle=True añade res['_partidas']: un registro por partida (para skyjo.graficos)."""
    n = len(agentes)
    nombres = _nombres(agentes)
    rng = random.Random(semilla)
    perms = list(itertools.permutations(range(n))) if n <= 4 else None
    if perms:
        rng.shuffle(perms)
    acc = {k: dict(partidas=0, victorias=0.0, puntos=0.0, posicion=0.0, columnas=0.0, ilegales=0) for k in nombres}
    truncadas, partidas = 0, []
    for g in range(n_partidas):
        p = perms[g % len(perms)] if perms else rng.sample(range(n), n)       # asiento s lo ocupa el agente p[s]
        r = jugar_partida([agentes[i] for i in p], motor, max_pasos)
        truncadas += r.truncada
        if detalle:
            partidas.append(dict(
                jugadores=[nombres[i] for i in p], puntos=list(r.puntos), posiciones=list(r.posiciones),
                pasos=r.pasos, truncada=r.truncada, penalizada=r.penalizada,
                cerrador=None if r.cerrador is None else nombres[p[r.cerrador]]))
        for s, i in enumerate(p):
            a = acc[nombres[i]]
            a['partidas'] += 1
            a['victorias'] += (1.0 / len(r.ganadores)) if s in r.ganadores else 0.0
            a['puntos'] += r.puntos[s]; a['posicion'] += r.posiciones[s]
            a['columnas'] += r.columnas[s]; a['ilegales'] += r.ilegales[s]
    res = {}
    for k, a in acc.items():
        m = max(a['partidas'], 1)
        res[k] = dict(partidas=a['partidas'], winrate=a['victorias'] / m, puntos=a['puntos'] / m,
                      posicion=a['posicion'] / m, columnas=a['columnas'] / m, ilegales=a['ilegales'])
    res['_global'] = dict(partidas=n_partidas, truncadas=truncadas, jugadores=n)
    if detalle:
        res['_partidas'] = partidas
    return res


def imprimir_evaluacion(res):
    g = res['_global']
    print(f"{g['partidas']} partidas, {g['jugadores']} jugadores" + (f"  ({g['truncadas']} truncadas)" if g['truncadas'] else ""))
    print(f"{'agente':<22}{'win%':>7}{'puntos':>8}{'pos.':>6}{'cols':>6}{'ilegales':>9}")
    for k, v in res.items():
        if not k.startswith('_'):
            print(f"{k:<22}{100 * v['winrate']:>7.1f}{v['puntos']:>8.1f}{v['posicion']:>6.2f}{v['columnas']:>6.2f}{v['ilegales']:>9}")
