"""Evalúa modelos ya entrenados contra heurísticos y muestra puntos + estadísticas de cierre de ronda.

  python evaluar_cierre.py --version v6_0 --modelo modelos/v6_0.pth --jugadores 2 3 4
  python evaluar_cierre.py --version v6_0 --ckpt-dir checkpoints_6.0 --jugadores 4 -n 600 --csv cierre.csv   # todos los ep*.pth y best.pth
  python evaluar_cierre.py --version v5_0 --modelo modelos/v5_6.pth --modelo modelos/v5_7.pth --jugadores 4

Cada modelo juega como 1 DQN contra (n-1) heurísticos con las mismas barajas (--semilla) en todos los asientos, igual que la
evaluación del entrenamiento. Columnas: puntos (± error típico) del DQN y de los heurísticos; 'cierra%' = fracción de rondas
cerradas que cierra cada uno (1/n sería igualdad); 'ocultas' = cartas boca abajo que le quedan cuando cierra otro;
'turnos' = turnos que tarda en cerrar cuando cierra él. Con --ckpt-dir ves cómo evoluciona el ritmo durante el entrenamiento.
Sirve para v5_0 y v6_0 (las versiones con evaluar_vs_heuristicos).
"""
import argparse
import csv
import glob
import importlib
import os
import re

import torch


def _checkpoints(ckpt_dir):
    eps = sorted(glob.glob(os.path.join(ckpt_dir, 'ep*.pth')), key=lambda p: int(re.findall(r'ep(\d+)\.pth', p)[0]))
    best = os.path.join(ckpt_dir, 'best.pth')
    return eps + ([best] if os.path.exists(best) else [])


def _f(x, fmt, factor=1.0):
    return '  n/d' if x is None else format(x * factor, fmt)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--version', required=True, help='v5_0 o v6_0: define la red y el agente')
    ap.add_argument('--modelo', action='append', default=[], help='fichero .pth con los pesos (repetible)')
    ap.add_argument('--ckpt-dir', default=None, help='carpeta con ep*.pth y best.pth: se evalúan todos')
    ap.add_argument('--jugadores', type=int, nargs='+', default=[4, 2])
    ap.add_argument('-n', type=int, default=300, help='partidas por modelo y nº de jugadores (= barajas x asientos)')
    ap.add_argument('--semilla', type=int, default=12345, help='mismas barajas para todos los modelos (comparación pareada)')
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    ap.add_argument('--csv', default=None, help='guarda todas las métricas en este CSV')
    a = ap.parse_args(argv)

    m = importlib.import_module(f'skyjo.{a.version}')
    ag = importlib.import_module(f'skyjo.{a.version}.agents')
    if not hasattr(ag, 'evaluar_vs_heuristicos'):
        raise SystemExit(f"{a.version} no tiene evaluar_vs_heuristicos: usa v5_0 o v6_0")
    rutas = list(a.modelo) + (_checkpoints(a.ckpt_dir) if a.ckpt_dir else [])
    if not rutas:
        raise SystemExit("indica --modelo y/o --ckpt-dir")

    filas = []
    print(f"{'modelo':<22}{'jug':>4}{'puntos':>14}{'(heur.)':>9}{'dif':>7}{'cierra%':>9}{'(heur.)':>9}{'ocultas':>9}{'(heur.)':>9}{'turnos':>8}{'(heur.)':>9}")
    for ruta in rutas:
        net = m.SkyjoDQN(2).to(a.device)
        net.load_state_dict(torch.load(ruta, map_location=a.device))
        net.eval()
        for n in a.jugadores:
            r = ag.evaluar_vs_heuristicos(net, a.device, n, a.n, semilla=a.semilla)
            d, h = r['_dqn'], r['heuristicos']
            print(f"{os.path.basename(ruta):<22}{n:>4}{d['puntos']:>8.2f}±{d['puntos_ee']:<5.2f}{h['puntos']:>9.2f}{d['puntos'] - h['puntos']:>+7.2f}"
                  f"{_f(d.get('cierra'), '9.1f', 100)}{_f(h.get('cierra'), '9.1f', 100)}{_f(d.get('ocultas_cierre'), '9.2f')}"
                  f"{_f(h.get('ocultas_cierre'), '9.2f')}{_f(d.get('turnos_cierra'), '8.1f')}{_f(h.get('turnos_cierra'), '9.1f')}")
            filas.append(dict(modelo=ruta, jugadores=n, partidas=r['_global']['partidas'], truncadas=r['_global']['truncadas'],
                              **{f'dqn_{k}': v for k, v in d.items() if k != 'partidas'},
                              **{f'heur_{k}': v for k, v in h.items() if k != 'partidas'}))
    if a.csv and filas:
        with open(a.csv, 'w', newline='', encoding='utf-8') as f:
            w = csv.DictWriter(f, fieldnames=list(filas[0]))
            w.writeheader()
            w.writerows(filas)
        print('->', a.csv)


if __name__ == '__main__':
    main()
