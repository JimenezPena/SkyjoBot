"""Agente DQN 5.0: decide solo a partir de la vista pública, con cualquier nº de jugadores (2-8)."""
import torch

from skyjo.agentes import Agente
from .observacion import observacion_desde_vista, a_tensores


class AgenteDQN(Agente):
    def __init__(self, model, device="cpu", nombre="dqn_v5_0"):
        self.model, self.device, self.nombre = model.eval(), device, nombre

    def decidir(self, vista):
        with torch.no_grad():
            q = self.model(a_tensores([observacion_desde_vista(vista)], self.device))
        q[0, ~torch.tensor(vista.mask, device=self.device)] = -1e9
        return int(q.argmax(dim=1).item())
