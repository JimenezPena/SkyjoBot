"""Rivales del entrenamiento: mezcla de heurísticos, la red actual y checkpoints antiguos.

Cada asiento que no aprende en una partida se sortea entre tres tipos (pesos en CONFIG['RIVALES']):
  'heuristico'  el bot heurístico (baseline)
  'actual'      la red que se está entrenando, jugando en modo greedy (con un poco de ruido: RIVAL_EPSILON)
  'checkpoint'  una copia congelada de la red de un momento anterior (se guarda una cada POOL_EVERY episodios; se
                conservan las POOL_MAX últimas) o un modelo externo de RIVALES_CKPT (p. ej. los pesos de la 5.1)
Si aún no hay checkpoints, 'checkpoint' se sustituye por 'heuristico'. Con RIVALES_FIN los pesos se interpolan
linealmente durante el entrenamiento (p. ej. de casi todo heurísticos a más self-play)."""
import random
from collections import deque

import torch

from skyjo.agentes import ConRuido
from skyjo.heuristico import AgenteHeuristico
from .agente import AgenteDQN
from .model import SkyjoDQN

TIPOS = ('heuristico', 'actual', 'checkpoint')


def pesos_en(config, progreso):
    """Pesos de cada tipo de rival en el instante `progreso` (0 = inicio, 1 = final del entrenamiento)."""
    ini = config['RIVALES']
    fin = config.get('RIVALES_FIN') or ini
    return {t: (1 - progreso) * ini.get(t, 0.0) + progreso * fin.get(t, 0.0) for t in TIPOS}


class PoolRivales:
    def __init__(self, policy_net, device, epsilon=0.05, max_checkpoints=5, rutas=()):
        self.device, self.epsilon = device, float(epsilon)
        self.heuristico = AgenteHeuristico()
        self.actual = ConRuido(AgenteDQN(policy_net, device, nombre="rival_actual"), self.epsilon)
        self.fijos = [self._agente(self._cargar_red(r), f"rival_{r}") for r in rutas]     # nunca se retiran
        self.congelados = deque(maxlen=int(max_checkpoints))                              # instantáneas propias

    def _cargar_red(self, ruta):
        net = SkyjoDQN().to(self.device)
        net.load_state_dict(torch.load(ruta, map_location=self.device))
        return net.eval()

    def _agente(self, net, nombre):
        return ConRuido(AgenteDQN(net, self.device, nombre=nombre), self.epsilon)

    def anadir_instantanea(self, policy_net, episodio=0):
        net = SkyjoDQN().to(self.device)
        net.load_state_dict(policy_net.state_dict())
        self.congelados.append(self._agente(net.eval(), f"rival_ep{episodio}"))

    def __len__(self):
        return len(self.fijos) + len(self.congelados)

    def elegir(self, pesos):
        """Agente rival sorteado según los pesos."""
        tipo = random.choices(TIPOS, weights=[max(pesos[t], 0.0) for t in TIPOS])[0] if any(pesos[t] > 0 for t in TIPOS) else 'heuristico'
        if tipo == 'checkpoint':
            miembros = self.fijos + list(self.congelados)
            return random.choice(miembros) if miembros else self.heuristico
        return self.actual if tipo == 'actual' else self.heuristico

    def guardar(self, ruta):
        torch.save([a.base.model.state_dict() for a in self.congelados], ruta)

    def cargar(self, ruta):
        for k, sd in enumerate(torch.load(ruta, map_location=self.device)):
            net = SkyjoDQN().to(self.device)
            net.load_state_dict(sd)
            self.congelados.append(self._agente(net.eval(), f"rival_cargado{k}"))
