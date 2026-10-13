"""Registro de niveles (niveles.json) y carga de agentes por nivel.

  "dqn_4.1":   {"tipo": "dqn", "version": "v4_2", "modelo": "modelos/v4_1.pth", "epsilon": 0.0}
  "heuristico": {"tipo": "heuristico"}      # baseline, sin pesos, válido con cualquier nº de jugadores
  "aleatorio":  {"tipo": "aleatorio"}

cargar_nivel(nombre) devuelve un agente f(env, obs, mask) -> acción válido con el entorno de cualquier versión.
"""
import importlib
import json
import os
import random

import torch

from .arena import AgenteSincronizado
from .heuristico import agente_aleatorio, agente_heuristico

RAIZ = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
RUTA_NIVELES = os.path.join(RAIZ, 'niveles.json')


def listar_niveles(ruta=None):
    with open(ruta or RUTA_NIVELES, encoding='utf-8') as f:
        return json.load(f)


def cargar_nivel(nombre, device="cpu", ruta=None):
    niveles = listar_niveles(ruta)
    if nombre not in niveles:
        raise KeyError(f"Nivel desconocido: '{nombre}'. Disponibles: {list(niveles)}")
    cfg = niveles[nombre]
    tipo = cfg['tipo']
    if tipo == 'aleatorio':
        base = agente_aleatorio
    elif tipo == 'heuristico':
        base = agente_heuristico
    elif tipo == 'dqn':
        modulo = importlib.import_module(f"skyjo.{cfg['version']}")
        path = cfg['modelo'] if os.path.isabs(cfg['modelo']) else os.path.join(RAIZ, cfg['modelo'])
        model = modulo.SkyjoDQN(2).to(device)
        try:
            model.load_state_dict(torch.load(path, map_location=device))
        except RuntimeError as e:
            raise ValueError(f"El modelo '{path}' no es compatible con la versión "
                             f"'{cfg['version']}' (nivel '{nombre}').") from e
        model.eval()
        base = AgenteSincronizado(modulo, modulo.agente_dqn(model, device))
    else:
        raise ValueError(f"Tipo de nivel desconocido: {tipo}")

    eps = float(cfg.get('epsilon', 0.0))
    if eps <= 0:
        return base

    def con_ruido(env, obs, mask):
        if random.random() < eps:
            return agente_aleatorio(env, obs, mask)
        return base(env, obs, mask)
    return con_ruido
