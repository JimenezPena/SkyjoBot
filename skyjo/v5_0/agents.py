"""Evaluación de v5.0: el modelo contra heurísticos con cualquier nº de asientos (rotando las posiciones)."""
from skyjo.heuristico import AgenteHeuristico
from skyjo.partida import evaluar, imprimir_evaluacion

from .agente import AgenteDQN


def evaluar_vs_heuristicos(net, device="cpu", n_jugadores=4, n_partidas=300, semilla=None):
    """1 DQN contra (n_jugadores-1) heurísticos. Con n asientos el azar da 100/n % de victorias."""
    dqn = AgenteDQN(net, device)
    res = evaluar([dqn] + [AgenteHeuristico() for _ in range(n_jugadores - 1)], n_partidas, semilla=semilla)
    res['_dqn'] = res[dqn.nombre]
    return res


def imprimir_vs_heuristicos(titulo, res):
    print(f"== {titulo} ==")
    imprimir_evaluacion(res)


def evaluar_final(net, device, n_partidas, n_jugadores=(4, 2)):
    """Evaluación final para train.py: devuelve {nº de asientos: métricas del DQN}."""
    out = {}
    for n in n_jugadores:
        r = evaluar_vs_heuristicos(net, device, n, n_partidas)
        imprimir_vs_heuristicos(f"FINAL {n} jugadores (1 DQN vs {n - 1} heurísticos)", r)
        out[str(n)] = r['_dqn']
    return out
