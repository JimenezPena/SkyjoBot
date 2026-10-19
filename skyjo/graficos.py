"""Resumen numérico e histogramas de una evaluación hecha con evaluar(..., detalle=True).

  res = evaluar(agentes, 500, detalle=True)
  imprimir_distribuciones(res['_partidas'])
  histogramas(res['_partidas'], 'plots')          # requiere matplotlib (opcional; no lo necesita la web)

Cada partida es un dict: jugadores (nombre por asiento), puntos, posiciones, pasos, truncada, penalizada, cerrador.
"""
import os
import statistics as st

# Okabe-Ito (apta para daltonismo), siempre en el mismo orden: el color sigue al agente.
COLORES = ['#0072B2', '#E69F00', '#009E73', '#CC79A7', '#56B4E9', '#D55E00', '#F0E442', '#000000']


def _por_agente(partidas):
    """{agente: dict(puntos, brecha, posicion, cerro, penalizada, gano_cerrando)} con una entrada por partida."""
    out = {}
    for g in partidas:
        mejor = min(g['puntos'])
        for s, nombre in enumerate(g['jugadores']):
            d = out.setdefault(nombre, dict(puntos=[], brecha=[], posicion=[], cerro=[], penalizada=[], gano=[]))
            d['puntos'].append(g['puntos'][s])
            d['brecha'].append(g['puntos'][s] - mejor)
            d['posicion'].append(g['posiciones'][s])
            cerro = g['cerrador'] == nombre
            d['cerro'].append(cerro)
            d['penalizada'].append(cerro and g['penalizada'])
            d['gano'].append(cerro and g['puntos'][s] == mejor)
    return out


def _pct(x):
    return 100 * sum(x) / max(len(x), 1)


def _cuantil(x, q):
    x = sorted(x)
    return x[min(len(x) - 1, int(q * len(x)))]


def imprimir_distribuciones(partidas):
    ag = _por_agente(partidas)
    pasos = [g['pasos'] for g in partidas]
    cerradas = [not g['truncada'] and g['cerrador'] is not None for g in partidas]
    print(f"\npartidas cerradas: {_pct(cerradas):.1f}%   jugadas por partida: mediana {st.median(pasos):.0f}, "
          f"p10 {_cuantil(pasos, .1)}, p90 {_cuantil(pasos, .9)}")
    print(f"{'agente':<22}{'media':>7}{'mediana':>8}{'desv.':>7}{'p90':>6}{'brecha':>8}{'cierra%':>8}{'cierra y dobla%':>16}{'cierra y gana%':>15}")
    for k, d in ag.items():
        n_c = sum(d['cerro'])
        print(f"{k:<22}{st.mean(d['puntos']):>7.1f}{st.median(d['puntos']):>8.1f}{st.pstdev(d['puntos']):>7.1f}"
              f"{_cuantil(d['puntos'], .9):>6}{st.mean(d['brecha']):>8.1f}{_pct(d['cerro']):>8.1f}"
              f"{100 * sum(d['penalizada']) / max(n_c, 1):>16.1f}{100 * sum(d['gano']) / max(n_c, 1):>15.1f}")
    print("(brecha = puntos propios − puntos del ganador; cierra y dobla/gana: % sobre las partidas que cerró)")


def histogramas(partidas, carpeta):
    """Guarda cinco PNG en `carpeta` y devuelve sus rutas."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np

    os.makedirs(carpeta, exist_ok=True)
    ag = _por_agente(partidas)
    col = {k: COLORES[i % len(COLORES)] for i, k in enumerate(ag)}
    n = len(next(iter(partidas))['jugadores'])
    rutas = []

    def guardar(fig, nombre):
        ruta = os.path.join(carpeta, nombre)
        fig.tight_layout()
        fig.savefig(ruta, dpi=130)
        plt.close(fig)
        rutas.append(ruta)

    def estilo(ax, titulo, x, y='% de partidas'):
        ax.set_title(titulo, loc='left', fontsize=11)
        ax.set_xlabel(x); ax.set_ylabel(y)
        for lado in ('top', 'right'):
            ax.spines[lado].set_visible(False)
        ax.grid(axis='y', alpha=.25)

    def hist(ax, clave, bins):
        for k, d in ag.items():
            w = np.full(len(d[clave]), 100 / len(d[clave]))
            ax.hist(d[clave], bins=bins, weights=w, histtype='stepfilled', alpha=.18, color=col[k])
            ax.hist(d[clave], bins=bins, weights=w, histtype='step', linewidth=1.8, color=col[k], label=k)
        ax.legend(frameon=False)

    todos = [x for d in ag.values() for x in d['puntos']]
    bins = np.arange(min(todos) - 2, max(todos) + 6, 4)
    fig, ax = plt.subplots(figsize=(8, 4.2))
    hist(ax, 'puntos', bins)
    for k, d in ag.items():
        ax.axvline(st.mean(d['puntos']), color=col[k], linestyle=':', linewidth=1)
    estilo(ax, 'Puntos finales por jugador (línea punteada: media)', 'puntos (menos es mejor)')
    guardar(fig, '1_puntos.png')

    fig, ax = plt.subplots(figsize=(8, 4.2))
    hist(ax, 'brecha', np.arange(0, max(x for d in ag.values() for x in d['brecha']) + 6, 4))
    estilo(ax, 'Puntos respecto al ganador de cada partida (0 = ganó)', 'puntos propios − puntos del ganador')
    guardar(fig, '2_brecha_al_ganador.png')

    fig, ax = plt.subplots(figsize=(8, 4.2))
    ancho = 0.8 / len(ag)
    for i, (k, d) in enumerate(ag.items()):
        h = [100 * sum(p == pos for p in d['posicion']) / len(d['posicion']) for pos in range(1, n + 1)]
        ax.bar(np.arange(1, n + 1) + (i - (len(ag) - 1) / 2) * ancho, h, ancho * .9, color=col[k], label=k)
    ax.set_xticks(range(1, n + 1)); ax.legend(frameon=False)
    estilo(ax, 'Posición final en el ranking (empates comparten posición)', 'posición (1 = mejor)')
    guardar(fig, '3_posicion.png')

    pasos = [g['pasos'] for g in partidas]
    fig, ax = plt.subplots(figsize=(8, 4.2))
    ax.hist(pasos, bins=30, weights=np.full(len(pasos), 100 / len(pasos)), color=COLORES[0], alpha=.85)
    ax.axvline(st.median(pasos), color='#444', linestyle=':')
    estilo(ax, f'Jugadas por partida (todos los jugadores; mediana {st.median(pasos):.0f})', 'jugadas (pasos)')
    guardar(fig, '4_jugadas.png')

    fig, ax = plt.subplots(figsize=(8, 4.2))
    cerro, dobla, gana = ([_pct(d[c]) for d in ag.values()] for c in ('cerro', 'penalizada', 'gano'))
    x = np.arange(len(ag))
    for j, (v, et, c) in enumerate(((cerro, '% de partidas que cierra', '#0072B2'),
                                    (dobla, '… y le doblan (no era el mejor)', '#D55E00'),
                                    (gana, '… y gana', '#009E73'))):
        ax.bar(x + (j - 1) * .27, v, .25, color=c, label=et)
    ax.set_xticks(x); ax.set_xticklabels(list(ag)); ax.legend(frameon=False)
    cerradas = _pct([not g['truncada'] and g['cerrador'] is not None for g in partidas])
    estilo(ax, f'Cierre de partida (partidas cerradas: {cerradas:.1f}%)', '', '% (sobre todas las partidas)')
    guardar(fig, '5_cierres.png')
    return rutas
