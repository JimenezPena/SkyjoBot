"""Entrena una versión:  python train.py --version v5_0 --out-dir /content/drive/MyDrive/skyjo
Sobrescribir hiperparámetros:  --set NUM_EPISODIOS=5000 REWARD_SCALE=1/25
Al terminar escribe <SAVE_PATH>.meta.json (versión, commit, CONFIG y evaluación vs heurístico)."""
import argparse
import ast
import importlib
import json
import os
import subprocess
import time

import torch


def _valor(s):
    try:
        return ast.literal_eval(s)
    except Exception:
        try:
            return eval(s, {"__builtins__": {}}, {})      # permite 1/25
        except Exception:
            return s


def _git(*args):
    try:
        return subprocess.check_output(["git", *args], stderr=subprocess.DEVNULL, text=True).strip()
    except Exception:
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--version', required=True, help='v4_2 | v5_0 ...')
    ap.add_argument('--out-dir', default='.', help='carpeta de salida (p. ej. en Drive)')
    ap.add_argument('--set', nargs='*', default=[], metavar='CLAVE=VALOR')
    ap.add_argument('--eval-games', type=int, default=300, help='partidas finales vs heurístico (0 = no)')
    args = ap.parse_args()

    m = importlib.import_module(f'skyjo.{args.version}')
    cfg = dict(m.CONFIG)
    for kv in args.set:
        k, v = kv.split('=', 1)
        if k not in cfg:
            raise SystemExit(f"CONFIG no tiene la clave '{k}'")
        cfg[k] = _valor(v)
    for k in ('SAVE_PATH', 'CKPT_DIR'):
        if k in cfg and not os.path.isabs(cfg[k]):
            cfg[k] = os.path.join(args.out_dir, cfg[k])
    os.makedirs(os.path.dirname(os.path.abspath(cfg['SAVE_PATH'])), exist_ok=True)

    t0 = time.time()
    m.entrenar_skyjo(cfg)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    meta = dict(version=args.version, commit=_git("describe", "--tags", "--always", "--dirty"),
                fecha=time.strftime("%Y-%m-%d %H:%M:%S"), minutos=round((time.time() - t0) / 60, 1),
                config=cfg)
    if args.eval_games > 0:
        net = m.SkyjoDQN(2).to(device)
        net.load_state_dict(torch.load(cfg['SAVE_PATH'], map_location=device))
        net.eval()
        r = m.duelo(m.agente_dqn(net, device), m.agente_heuristico, args.eval_games)
        m.imprimir_duelo("FINAL vs heurístico", r)
        meta['eval_vs_heuristico'] = r
    with open(os.path.splitext(cfg['SAVE_PATH'])[0] + '.meta.json', 'w', encoding='utf-8') as f:
        json.dump(meta, f, indent=2, ensure_ascii=False, default=float)


if __name__ == "__main__":
    main()
