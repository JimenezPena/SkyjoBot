"""Skyjo completo: se juegan rondas hasta que alguien llega a LIMITE (100) puntos o más; gana quien menos suma.

  from skyjo.match import evaluar_matches, imprimir_matches
  imprimir_matches(evaluar_matches([agente_a, agente_b, agente_c, agente_d], n_matches=50))

Cada ronda es una partida de skyjo.partida.jugar_partida (con la penalización al cerrador). Los asientos rotan
igual que en evaluar(); los empates a puntos reparten la victoria.
"""
import itertools
import random

from .partida import jugar_partida, _nombres


def jugar_match(agentes, limite=100, motor=None, max_pasos=3000, max_rondas=100):
    """Devuelve dict(total=[puntos acumulados por asiento], rondas, ganadores, truncado)."""
    n = len(agentes)
    total, rondas = [0] * n, 0
    while max(total) < limite and rondas < max_rondas:
        r = jugar_partida(agentes, motor, max_pasos)
        if r.truncada:
            return dict(total=total, rondas=rondas, ganadores=[], truncado=True)
        total = [t + p for t, p in zip(total, r.puntos)]
        rondas += 1
    mejor = min(total)
    return dict(total=total, rondas=rondas, ganadores=[k for k, t in enumerate(total) if t == mejor],
                truncado=rondas >= max_rondas and max(total) < limite)


def evaluar_matches(agentes, n_matches=50, limite=100, motor=None, max_pasos=3000, semilla=None):
    n = len(agentes)
    nombres = _nombres(agentes)
    rng = random.Random(semilla)
    perms = list(itertools.permutations(range(n))) if n <= 4 else None
    if perms:
        rng.shuffle(perms)
    acc = {k: dict(m=0, victorias=0.0, total=0.0) for k in nombres}
    rondas, truncados = [], 0
    for g in range(n_matches):
        p = perms[g % len(perms)] if perms else rng.sample(range(n), n)       # asiento s lo ocupa el agente p[s]
        r = jugar_match([agentes[i] for i in p], limite, motor, max_pasos)
        if r['truncado']:
            truncados += 1
            continue
        rondas.append(r['rondas'])
        for s, i in enumerate(p):
            a = acc[nombres[i]]
            a['m'] += 1
            a['victorias'] += (1.0 / len(r['ganadores'])) if s in r['ganadores'] else 0.0
            a['total'] += r['total'][s]
    res = {k: dict(matches=a['m'], winrate=a['victorias'] / max(a['m'], 1), puntos=a['total'] / max(a['m'], 1))
           for k, a in acc.items()}
    res['_global'] = dict(matches=n_matches, truncados=truncados, jugadores=n, limite=limite,
                          rondas=sum(rondas) / max(len(rondas), 1))
    return res


def imprimir_matches(res):
    g = res['_global']
    print(f"{g['matches']} matches a {g['limite']} puntos, {g['jugadores']} jugadores, {g['rondas']:.1f} rondas de media"
          + (f"  ({g['truncados']} truncados)" if g['truncados'] else ""))
    print(f"{'agente':<22}{'win%':>7}{'puntos finales':>16}")
    for k, v in res.items():
        if not k.startswith('_'):
            print(f"{k:<22}{100 * v['winrate']:>7.1f}{v['puntos']:>16.1f}")
