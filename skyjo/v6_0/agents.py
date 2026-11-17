"""Evaluación de v5.0: el modelo contra heurísticos con cualquier nº de asientos.

Cada baraja se juega n veces, con el DQN en cada uno de los n asientos: recibe así cada mano inicial de la misma baraja
y los asientos quedan equilibrados. Misma semilla = mismas barajas, así que la comparación entre checkpoints es pareada.
Métrica principal: puntuación media final del DQN por partida (menos es mejor)."""
import random

from skyjo.heuristico import AgenteHeuristico
from skyjo.partida import jugar_partida, imprimir_evaluacion

from .agente import AgenteDQN


def _acum():
    return dict(partidas=0, victorias=0.0, puntos=0.0, puntos2=0.0, posicion=0.0, columnas=0.0, ilegales=0,
                cerradas=0, cierra=0, turnos_cierra=0.0, ocultas=0.0, n_ocultas=0)


def _sumar(a, r, k):
    a['partidas'] += 1
    a['victorias'] += (1.0 / len(r.ganadores)) if k in r.ganadores else 0.0
    a['puntos'] += r.puntos[k]
    a['puntos2'] += r.puntos[k] ** 2
    a['posicion'] += r.posiciones[k]
    a['columnas'] += r.columnas[k]
    a['ilegales'] += r.ilegales[k]
    if r.ocultas_cierre is not None:                      # la ronda llegó a cerrarse
        a['cerradas'] += 1
        if r.cerrador == k:
            a['cierra'] += 1
            a['turnos_cierra'] += r.turnos_cierre
        else:
            a['ocultas'] += r.ocultas_cierre[k]            # cartas que le quedaban boca abajo cuando otro cerró
            a['n_ocultas'] += 1


def _resumen(a):
    m = max(a['partidas'], 1)
    media = a['puntos'] / m
    var = max(a['puntos2'] / m - media ** 2, 0.0)
    return dict(partidas=a['partidas'], winrate=a['victorias'] / m, puntos=media, puntos_ee=(var / m) ** 0.5,
                posicion=a['posicion'] / m, columnas=a['columnas'] / m, ilegales=a['ilegales'],
                cierra=a['cierra'] / a['cerradas'] if a['cerradas'] else None,                  # fracción de rondas cerradas que cierra
                ocultas_cierre=a['ocultas'] / a['n_ocultas'] if a['n_ocultas'] else None,         # ocultas medias cuando cierra otro
                turnos_cierra=a['turnos_cierra'] / a['cierra'] if a['cierra'] else None)          # turnos que tarda cuando cierra él


def evaluar_vs_heuristicos(net, device="cpu", n_jugadores=4, n_partidas=300, semilla=None):
    """1 DQN contra (n_jugadores-1) heurísticos. Juega n_partidas // n_jugadores barajas (mínimo 1), cada una con el DQN
    en todos los asientos, así que n_partidas efectivas = barajas * n_jugadores.
    res['_dqn'] = métricas del DQN; res['heuristicos'] = media de los rivales. 'puntos_ee' es el error típico de la media
    de puntos (sin emparejar: orientativo)."""
    n = n_jugadores
    dqn = AgenteDQN(net, device)
    barajas = max(n_partidas // n, 1)
    rng = random.Random(semilla)
    semillas = [rng.randrange(2 ** 31) for _ in range(barajas)]
    acc_dqn, acc_heur, truncadas = _acum(), _acum(), 0
    for b in semillas:
        for s in range(n):
            agentes = [AgenteHeuristico() for _ in range(n - 1)]
            agentes.insert(s, dqn)
            r = jugar_partida(agentes, semilla=b)
            truncadas += r.truncada
            for k in range(n):
                _sumar(acc_dqn if k == s else acc_heur, r, k)
    res = {dqn.nombre: _resumen(acc_dqn), 'heuristicos': _resumen(acc_heur)}
    res['_dqn'] = res[dqn.nombre]
    res['_global'] = dict(partidas=barajas * n, truncadas=truncadas, jugadores=n)
    return res


def _linea_cierre(d, h):
    def f(x, fmt, suf=''):
        return 'n/d' if x is None else format(x, fmt) + suf
    return (f"cierre: DQN cierra {f(d.get('cierra') and 100 * d['cierra'], '.1f', '%')} (heurístico {f(h.get('cierra') and 100 * h['cierra'], '.1f', '%')} c/u) | "
            f"ocultas cuando cierra otro: DQN {f(d.get('ocultas_cierre'), '.2f')} (heur. {f(h.get('ocultas_cierre'), '.2f')}) | "
            f"turnos hasta cerrar: DQN {f(d.get('turnos_cierra'), '.1f')} (heur. {f(h.get('turnos_cierra'), '.1f')})")


def imprimir_vs_heuristicos(titulo, res):
    print(f"== {titulo} ==")
    imprimir_evaluacion(res)
    d = res['_dqn']
    print(f"puntos medios DQN: {d['puntos']:.2f} ± {d['puntos_ee']:.2f}  (heurísticos {res['heuristicos']['puntos']:.2f})")
    print(_linea_cierre(d, res['heuristicos']))


def evaluar_final(net, device, n_partidas, n_jugadores=(4, 2), semilla=None):
    """Evaluación final para train.py: devuelve {nº de asientos: métricas del DQN}."""
    out = {}
    for n in n_jugadores:
        r = evaluar_vs_heuristicos(net, device, n, n_partidas, semilla=semilla)
        imprimir_vs_heuristicos(f"FINAL {n} jugadores (1 DQN vs {n - 1} heurísticos)", r)
        out[str(n)] = r['_dqn']
    return out
