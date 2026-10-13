"""Agente DQN 4.x: decide solo a partir de la vista pública."""
import torch

from skyjo.agentes import Agente
from .observacion import observacion_desde_vista


class AgenteDQN(Agente):
    def __init__(self, model, device="cpu", nombre="dqn_v4_2"):
        self.model, self.device, self.nombre = model.eval(), device, nombre

    def decidir(self, vista):
        if vista.n != self.model.num_jugadores:
            raise ValueError(f"Este modelo solo juega con {self.model.num_jugadores} jugadores (la vista tiene {vista.n}).")
        obs = observacion_desde_vista(vista)
        with torch.no_grad():
            q = self.model({k: torch.from_numpy(v).unsqueeze(0).to(self.device) for k, v in obs.items()})
        mask = torch.tensor(vista.mask, device=self.device)
        q[0, ~mask] = -1e9
        return int(q.argmax(dim=1).item())
