import torch
import torch.nn as nn

from .env import N_ACCIONES
from .observacion import N_ESCALARES

H = 64            # tamaño del embedding de cada carta (nodo)
CAPAS = 3         # rondas de paso de mensajes dentro del tablero


def _nodos(x):
    """[M, 2, 3, 4] (valor normalizado, estado) -> [M, 3, 4, 4]: valor | visible | oculta | eliminada."""
    val, est = x[:, 0], x[:, 1]
    return torch.stack([val, (est == 1).float(), (est == 0).float(), (est == -1).float()], dim=-1)


class CapaColumna(nn.Module):
    """Una ronda de paso de mensajes sobre un tablero 3x4: cada carta recibe un mensaje de las otras 2 de SU columna
    (arista = misma columna, que es donde está la regla de eliminar tríos) y, opcionalmente, del vector de contexto g
    (un nodo global conectado a las 12 cartas). El mensaje es una MLP sobre el par (yo, vecina) y se suma, así que la
    capa es equivariante a permutar filas y columnas: ambas son simetrías reales del juego."""
    def __init__(self, h):
        super().__init__()
        self.propio = nn.Linear(h, h)
        self.par = nn.Sequential(nn.Linear(2 * h, h), nn.ReLU(), nn.Linear(h, h))
        self.ctx = nn.Linear(h, h, bias=False)
        self.norma = nn.LayerNorm(h)

    def forward(self, x, g=None):                       # x: [M, 3, 4, h]; g: [M, h] o None
        # las 2 vecinas de la columna (filas desplazadas 1 y 2) en una sola llamada: menos operaciones por paso
        vecinas = torch.stack([torch.roll(x, 1, dims=1), torch.roll(x, 2, dims=1)])          # [2, M, 3, 4, h]
        msg = self.par(torch.cat([x.expand_as(vecinas), vecinas], -1)).sum(dim=0)
        z = self.propio(x) + msg
        if g is not None:
            z = z + self.ctx(g)[:, None, None, :]
        return self.norma(x + torch.relu(z))            # conexión residual


class SkyjoDQN(nn.Module):
    """Dueling DQN v6.0: red de grafos sobre las 12 cartas del tablero. Mismas observación y entrenamiento que v5.0
    (solo cambia la red), para poder comparar. Parámetros independientes del nº de jugadores.

    - Cada tablero (yo, izquierda, derecha y los 'otros') se procesa con la MISMA pila de CapaColumna y se resume con
      media y máximo sobre sus 12 cartas (invariante a permutaciones); cada rol tiene su proyección, y los 'otros' se
      resumen otra vez con media y máximo (invariante al orden y al número).
    - Con ese resumen y los escalares se forma un contexto g, que entra en las capas de MI tablero.
    - Cabezas por carta: Q(colocar en la casilla i) y Q(descartar y destapar la casilla i) salen del embedding de cada
      carta, así que lo aprendido en una casilla vale para todas (acciones 2-13 y 14-25, orden por filas f*4+c).
      Robar (acciones 0-1) y el valor de estado salen del resumen global.
    """
    def __init__(self, num_jugadores=None, h=H, capas=CAPAS):
        super().__init__()
        self.num_jugadores = None        # None = cualquier nº (2-8); el argumento existe por compatibilidad con niveles.py
        self.embed = nn.Linear(4, h)
        self.capas = nn.ModuleList(CapaColumna(h) for _ in range(capas))
        self.p_izq, self.p_der, self.p_otros = (nn.Linear(2 * h, h) for _ in range(3))
        self.fc_escalares = nn.Sequential(nn.Linear(N_ESCALARES, 64), nn.ReLU())
        self.fc_ctx = nn.Sequential(nn.Linear(64 + 4 * h, h), nn.ReLU())
        self.q_colocar = nn.Sequential(nn.Linear(2 * h, h), nn.ReLU(), nn.Linear(h, 1))
        self.q_destapar = nn.Sequential(nn.Linear(2 * h, h), nn.ReLU(), nn.Linear(h, 1))
        self.q_robar = nn.Sequential(nn.Linear(3 * h, h), nn.ReLU(), nn.Linear(h, 2))
        self.valor = nn.Sequential(nn.Linear(3 * h, h), nn.ReLU(), nn.Linear(h, 1))

    def _grafo(self, x, g=None):         # x: [M, 2, 3, 4] -> [M, 3, 4, h]
        h = self.embed(_nodos(x))
        for capa in self.capas:
            h = capa(h, g)
        return h

    @staticmethod
    def _resumen(h):                     # [M, 3, 4, h] -> [M, 2h]
        n = h.flatten(1, 2)
        return torch.cat([n.mean(dim=1), n.max(dim=1).values], dim=1)

    def forward(self, obs):
        B = obs['escalares'].size(0)
        # los tableros rivales (izq, der y los 'otros') pasan por el grafo juntos, en un solo batch
        rivales = torch.cat([obs['tablero_izq'][:, None], obs['tablero_der'][:, None], obs['tableros_otros']], dim=1)
        e = self._resumen(self._grafo(rivales.flatten(0, 1))).view(B, rivales.size(1), -1)      # [B, 2+k, 2h]
        izq = torch.relu(self.p_izq(e[:, 0]))
        der = torch.relu(self.p_der(e[:, 1]))
        if e.size(1) > 2:                                       # k = n-3 'otros' (puede ser 0)
            o = torch.relu(self.p_otros(e[:, 2:]))
            pool = torch.cat([o.mean(dim=1), o.max(dim=1).values], dim=1)
        else:
            pool = izq.new_zeros(B, 2 * izq.size(1))
        g = self.fc_ctx(torch.cat([self.fc_escalares(obs['escalares']), izq, der, pool], dim=1))

        nodos = self._grafo(obs['tablero_yo'], g).flatten(1, 2)         # [B, 12, h], orden por filas = casilla f*4+c
        por_carta = torch.cat([nodos, g[:, None, :].expand(-1, nodos.size(1), -1)], dim=-1)
        glob = torch.cat([nodos.mean(dim=1), nodos.max(dim=1).values, g], dim=1)
        a = torch.cat([self.q_robar(glob), self.q_colocar(por_carta).squeeze(-1), self.q_destapar(por_carta).squeeze(-1)], dim=1)
        assert a.size(1) == N_ACCIONES
        return self.valor(glob) + (a - a.mean(dim=1, keepdim=True))
