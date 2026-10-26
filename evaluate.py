"""Evaluación.
  python evaluate.py torneo --version v5_0 --ckpt-dir checkpoints_5.1 --jugadores 4   # cada checkpoint vs heurísticos
  python evaluate.py niveles -n 300                                           # niveles de dos en dos (niveles.json)
  python evaluate.py partida -n 240 heuristico aleatorio heuristico aleatorio # N asientos, rotando asientos
  python evaluate.py match -n 100 dqn_5.1 heuristico heuristico heuristico    # juego completo: rondas hasta 100 puntos
  python evaluate.py partida -n 1000 --plots plots dqn_5.1 heuristico heuristico heuristico   # + histogramas (matplotlib)
"""
import argparse
import glob
import importlib
import os
import re

import torch

from skyjo.niveles import cargar_nivel, listar_niveles
from skyjo.match import evaluar_matches, imprimir_matches
from skyjo.partida import evaluar, imprimir_evaluacion
from skyjo.graficos import histogramas, imprimir_distribuciones


def torneo(version, ckpt_dir, n, device, jugadores=2, semilla=0):
    """Cada checkpoint (ep*.pth y best.pth) como agente 0 contra heurísticos en `jugadores` asientos."""
    from skyjo.heuristico import AgenteAleatorio, AgenteHeuristico
    from skyjo.partida import _nombres
    m = importlib.import_module(f'skyjo.{version}')

    def cargar(p):
        net = m.SkyjoDQN(2).to(device)
        net.load_state_dict(torch.load(p, map_location=device))
        return m.AgenteDQN(net.eval(), device)

    def duelo(etiqueta, agente, rivales):
        ags = [agente] + rivales
        r = evaluar(ags, n, semilla=semilla)
        v = r[_nombres(ags)[0]]
        print(f"{etiqueta:<28}win {100 * v['winrate']:5.1f}%  puntos {v['puntos']:5.1f}  pos {v['posicion']:.2f}  "
              f"cols {v['columnas']:.2f}  ilegales {v['ilegales']}")

    heur = lambda: [AgenteHeuristico() for _ in range(jugadores - 1)]
    ckpts = sorted(glob.glob(f"{ckpt_dir}/ep*.pth"), key=lambda p: int(re.findall(r'ep(\d+)\.pth', p)[0]))
    best = f"{ckpt_dir}/best.pth"
    print(f"== Cada checkpoint vs heurístico ({jugadores} jugadores, {n} partidas; {100 / jugadores:.0f}% = paridad) ==")
    for p in ckpts + ([best] if os.path.exists(best) else []):
        duelo(os.path.basename(p), cargar(p), heur())
    if os.path.exists(best):
        print("\n== best vs aleatorio ==")
        duelo("best vs aleatorio", cargar(best), [AgenteAleatorio() for _ in range(jugadores - 1)])
        if len(ckpts) >= 2:
            print("\n== best vs primer y último checkpoint (resto: heurísticos) ==")
            for p in (ckpts[0], ckpts[-1]):
                duelo(f"best vs {os.path.basename(p)}", cargar(best), [cargar(p)] + heur()[1:])


def niveles(n, device, ruta, semilla=None):
    nombres = list(listar_niveles(ruta))
    for i, a in enumerate(nombres):
        for b in nombres[i + 1:]:
            print(f"\n== {a} vs {b} ==")
            imprimir_evaluacion(evaluar([cargar_nivel(a, device, ruta), cargar_nivel(b, device, ruta)], n, semilla=semilla))


def partida(n, device, asientos, ruta, plots=None, semilla=None):
    res = evaluar([cargar_nivel(a, device, ruta) for a in asientos], n, detalle=plots is not None, semilla=semilla)
    imprimir_evaluacion(res)
    if plots is not None:
        imprimir_distribuciones(res['_partidas'])
        print("\nHistogramas:", *histogramas(res['_partidas'], plots), sep="\n  ")


def match(n, device, asientos, ruta, limite, semilla=None):
    imprimir_matches(evaluar_matches([cargar_nivel(a, device, ruta) for a in asientos], n, limite, semilla=semilla))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest='cmd', required=True)
    t = sub.add_parser('torneo')
    t.add_argument('--version', required=True)
    t.add_argument('--ckpt-dir', required=True)
    t.add_argument('-n', type=int, default=500)
    t.add_argument('--semilla', type=int, default=0, help='mismos repartos para todos los checkpoints (comparación pareada)')
    t.add_argument('--jugadores', type=int, default=2, help='asientos de cada partida (rivales: heurísticos)')
    for nombre in ('niveles', 'partida', 'match'):
        s = sub.add_parser(nombre)
        s.add_argument('-n', type=int, default=300)
        s.add_argument('--semilla', type=int, default=None, help='mismos repartos en cada ejecución (comparación pareada)')
        s.add_argument('--niveles', default=None, help='ruta a un niveles.json alternativo')
        if nombre == 'match':
            s.add_argument('--limite', type=int, default=100, help='el match acaba al llegar alguien a estos puntos')
            s.add_argument('asientos', nargs='+', help='nombres de nivel, uno por asiento (2-8)')
        if nombre == 'partida':
            s.add_argument('--plots', nargs='?', const='plots', default=None, metavar='CARPETA',
                           help='guarda histogramas (PNG) y distribuciones; requiere matplotlib (por defecto ./plots)')
            s.add_argument('asientos', nargs='+', help='nombres de nivel, uno por asiento (2-8)')
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    if a.cmd == 'torneo':
        torneo(a.version, a.ckpt_dir, a.n, dev, a.jugadores, a.semilla)
    elif a.cmd == 'niveles':
        niveles(a.n, dev, a.niveles, a.semilla)
    elif a.cmd == 'match':
        match(a.n, dev, a.asientos, a.niveles, a.limite, a.semilla)
    else:
        partida(a.n, dev, a.asientos, a.niveles, a.plots, a.semilla)
