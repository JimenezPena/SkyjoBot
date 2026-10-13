"""Evaluación.
  python evaluate.py torneo --version v4_2 --ckpt-dir checkpoints_4.2        # cada checkpoint vs heurístico
  python evaluate.py niveles -n 300                                           # niveles de dos en dos (niveles.json)
  python evaluate.py partida -n 240 heuristico aleatorio heuristico aleatorio # N asientos, rotando asientos
"""
import argparse
import glob
import importlib
import re

import torch

from skyjo.niveles import cargar_nivel, listar_niveles
from skyjo.partida import evaluar, imprimir_evaluacion


def torneo(version, ckpt_dir, n, device):
    m = importlib.import_module(f'skyjo.{version}')
    imprimir_duelo = m.imprimir_duelo          # cada versión imprime sus propias métricas

    def cargar(p):
        net = m.SkyjoDQN(2).to(device)
        net.load_state_dict(torch.load(p, map_location=device))
        return net.eval()

    ckpts = sorted(glob.glob(f"{ckpt_dir}/ep*.pth"), key=lambda p: int(re.findall(r'ep(\d+)\.pth', p)[0]))
    best = f"{ckpt_dir}/best.pth"
    print(f"== Cada checkpoint vs heurístico ({n} partidas) ==")
    for p in ckpts + [best]:
        imprimir_duelo(p.split('/')[-1], m.duelo(m.agente_dqn(cargar(p), device), m.agente_heuristico, n))
    mb = cargar(best)
    print("\n== best vs aleatorio ==")
    imprimir_duelo("best vs aleatorio", m.duelo(m.agente_dqn(mb, device), m.agente_aleatorio, n))
    if len(ckpts) >= 2:
        print("\n== best vs primer y último checkpoint ==")
        for p in (ckpts[0], ckpts[-1]):
            imprimir_duelo(f"best vs {p.split('/')[-1]}",
                           m.duelo(m.agente_dqn(mb, device), m.agente_dqn(cargar(p), device), n))


def niveles(n, device, ruta):
    nombres = list(listar_niveles(ruta))
    for i, a in enumerate(nombres):
        for b in nombres[i + 1:]:
            print(f"\n== {a} vs {b} ==")
            imprimir_evaluacion(evaluar([cargar_nivel(a, device, ruta), cargar_nivel(b, device, ruta)], n))


def partida(n, device, asientos, ruta):
    imprimir_evaluacion(evaluar([cargar_nivel(a, device, ruta) for a in asientos], n))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest='cmd', required=True)
    t = sub.add_parser('torneo')
    t.add_argument('--version', required=True)
    t.add_argument('--ckpt-dir', required=True)
    t.add_argument('-n', type=int, default=500)
    for nombre in ('niveles', 'partida'):
        s = sub.add_parser(nombre)
        s.add_argument('-n', type=int, default=300)
        s.add_argument('--niveles', default=None, help='ruta a un niveles.json alternativo')
        if nombre == 'partida':
            s.add_argument('asientos', nargs='+', help='nombres de nivel, uno por asiento (2-8)')
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    if a.cmd == 'torneo':
        torneo(a.version, a.ckpt_dir, a.n, dev)
    elif a.cmd == 'niveles':
        niveles(a.n, dev, a.niveles)
    else:
        partida(a.n, dev, a.asientos, a.niveles)
