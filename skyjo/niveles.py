"""Registro de niveles (niveles.json) y carga de agentes por nivel.

  "dqn_4.2":    {"tipo": "dqn", "version": "v4_2", "modelo": "modelos/v4_2.pth", "epsilon": 0.0}
  "heuristico": {"tipo": "heuristico"}      # baseline, sin pesos, válido con cualquier nº de jugadores
  "aleatorio":  {"tipo": "aleatorio"}
  "llm_falso":  {"tipo": "llm", "cliente": "falso", "nivel_base": "dqn_5.1", "razonar": false, "oculto": true}   # ver skyjo/llm.py

cargar_nivel(nombre) devuelve un agente de vista: agente(vista) -> acción.
"""
import importlib
import json
import os

import torch

from .agentes import ConRuido
from .heuristico import AgenteAleatorio, AgenteHeuristico

RAIZ = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
RUTA_NIVELES = os.path.join(RAIZ, 'niveles.json')


def listar_niveles(ruta=None, ocultos=False):
    """Niveles de niveles.json. Los marcados con "oculto": true (p. ej. pruebas del agente LLM) no salen en la web
    ni en evaluate.py niveles, pero se pueden cargar por nombre."""
    with open(ruta or RUTA_NIVELES, encoding='utf-8') as f:
        d = json.load(f)
    return d if ocultos else {k: v for k, v in d.items() if not v.get('oculto')}


def jugadores_soportados(cfg):
    """(mínimo, máximo) de asientos que admite un nivel. Opcional en niveles.json: "jugadores": [min, max]."""
    if 'jugadores' in cfg:
        return tuple(cfg['jugadores'])
    if cfg['tipo'] == 'dqn':
        return tuple(getattr(importlib.import_module(f"skyjo.{cfg['version']}"), 'JUGADORES', (2, 2)))
    return (2, 8)


def niveles_para(n, ruta=None):
    """Nombres de los niveles que pueden jugar una partida de n asientos."""
    return [k for k, cfg in listar_niveles(ruta).items() if jugadores_soportados(cfg)[0] <= n <= jugadores_soportados(cfg)[1]]


def cargar_nivel(nombre, device="cpu", ruta=None):
    niveles = listar_niveles(ruta, ocultos=True)
    if nombre not in niveles:
        raise KeyError(f"Nivel desconocido: '{nombre}'. Disponibles: {list(niveles)}")
    cfg = niveles[nombre]
    tipo = cfg['tipo']
    if tipo == 'aleatorio':
        base = AgenteAleatorio()
    elif tipo == 'heuristico':
        base = AgenteHeuristico()
    elif tipo == 'dqn':
        modulo = importlib.import_module(f"skyjo.{cfg['version']}")
        path = cfg['modelo'] if os.path.isabs(cfg['modelo']) else os.path.join(RAIZ, cfg['modelo'])
        model = modulo.SkyjoDQN(2).to(device)
        try:
            model.load_state_dict(torch.load(path, map_location=device))
        except RuntimeError as e:
            raise ValueError(f"El modelo '{path}' no es compatible con la versión "
                             f"'{cfg['version']}' (nivel '{nombre}').") from e
        base = modulo.AgenteDQN(model.eval(), device)
    elif tipo == 'llm':
        from .llm import crear_agente_llm
        base = crear_agente_llm(cfg, device, ruta)
    else:
        raise ValueError(f"Tipo de nivel desconocido: {tipo}")
    base.nombre = nombre
    eps = float(cfg.get('epsilon', 0.0))
    return ConRuido(base, eps) if eps > 0 else base
