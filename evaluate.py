"""Evaluación.
  python evaluate.py torneo --version v5_0 --ckpt-dir checkpoints_5.0   # cada checkpoint vs heurístico
  python evaluate.py niveles -n 300                                      # todos contra todos (niveles.json)
"""
import argparse
import glob
import importlib
import re

import torch

from skyjo.niveles import cargar_nivel, listar_niveles
from skyjo.v5_0.agents import duelo, imprimir_duelo      # entorno árbitro (las reglas son iguales en todas las versiones)


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
    ag = {k: cargar_nivel(k, device, ruta) for k in nombres}
    print(f"== Todos contra todos ({n} partidas por duelo, alternando asiento) ==")
    for i, a in enumerate(nombres):
        for b in nombres[i + 1:]:
            imprimir_duelo(f"{a} vs {b}", duelo(ag[a], ag[b], n))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest='cmd', required=True)
    t = sub.add_parser('torneo')
    t.add_argument('--version', required=True)
    t.add_argument('--ckpt-dir', required=True)
    t.add_argument('-n', type=int, default=500)
    s = sub.add_parser('niveles')
    s.add_argument('-n', type=int, default=300)
    s.add_argument('--niveles', default=None, help='ruta a un niveles.json alternativo')
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    if a.cmd == 'torneo':
        torneo(a.version, a.ckpt_dir, a.n, dev)
    else:
        niveles(a.n, dev, a.niveles)
