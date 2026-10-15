"""Pre-entrenamiento por imitación (behaviour cloning) del heurístico.

1. Juega IMIT_EPISODIOS partidas heurístico vs heurístico con el propio entorno.
2. Rellena el ReplayBuffer con esas transiciones (con las mismas recompensas de
   shaping y final que usa el entrenamiento normal): así la fase DQN arranca con
   un buffer lleno de experiencia decente (idea central de DQfD).
3. Entrena la red con cross-entropy sobre las acciones del heurístico, con las
   acciones ilegales enmascaradas.

Detalles:
  - partidas de JUGADORES asientos.
  - la observación se construye desde la vista de cada asiento (como el trainer de 5.0),
  - la recompensa final es RECOMPENSA_FINAL * (media de rivales - mis puntos),
  - las demos se juegan con el mismo nº de jugadores que el entrenamiento, porque
    el ReplayBuffer fija sus formas desde una observación de ejemplo.
"""
import numpy as np
import torch
import torch.nn.functional as F

from skyjo.heuristico import AgenteHeuristico
from skyjo.vista import construir_vista
from .env import SkyjoEnv, N_ACCIONES
from .observacion import observacion_desde_vista, a_tensores


class FuenteDemos:
    """Guarda las demos del BC para reutilizarlas en la pérdida de margen de la fase DQN."""
    def __init__(self, demos, acciones, masks):
        self.demos, self.acciones, self.masks = demos, acciones, masks

    def sample(self, batch_size, device):
        i = np.random.randint(0, len(self.acciones), size=batch_size)
        obs = {k: torch.from_numpy(v[i]).to(device) for k, v in self.demos.items()}
        a = torch.from_numpy(self.acciones[i]).to(device)
        m = torch.from_numpy(self.masks[i]).to(device)
        return obs, a, m


def paso_margen(policy_net, fuente, optimizer, config, device):
    """Una actualización con la pérdida de margen DQfD sobre un batch de demos:
       L = mean(relu(m - (Q(s,a*) - max_{a legal, a!=a*} Q(s,a))))
       Se llama además del entrenar_step normal, mientras dure DEMO_STEPS."""
    obs, a, m = fuente.sample(config['DEMO_BATCH'], device)
    q = policy_net(obs)
    q_legales = q.masked_fill(~m, -1e9)
    q_demo = q_legales.gather(1, a.unsqueeze(1)).squeeze(1)
    
    # Clonamos antes de la operación in-place para evitar el error de versión de PyTorch
    q_legales_max = q_legales.clone()
    q_legales_max.scatter_(1, a.unsqueeze(1), -1e9)      # excluye la acción demo del máximo
    
    loss = F.relu(config['DEMO_MARGIN'] - (q_demo - q_legales_max.max(dim=1).values)).mean()

    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(policy_net.parameters(), max_norm=10.0)
    optimizer.step()
    return loss.item()
    
    
def _recompensas_finales(env, coef):
    """finales[j] = coef * (media de puntos de los rivales - mis puntos). Igual que trainer.py."""
    puntos = np.asarray(env.calcular_puntuaciones_finales(), dtype=np.float64)
    n = env.num_jugadores
    media_riv = (puntos.sum() - puntos) / (n - 1)
    return coef * (media_riv - puntos)


def _partidas_heuristicas(env, replay_buffer, config):
    """Partidas con JUGADORES heurísticos: demos para BC + transiciones al buffer."""
    n = env.num_jugadores
    heur = AgenteHeuristico()
    obs_list, acciones, masks = [], [], []

    for ep in range(1, config['IMIT_EPISODIOS'] + 1):
        env.reset()
        done, pendientes = False, {}
        vacia = np.zeros(N_ACCIONES, dtype=bool)

        while not done:
            j = env.turno_actual
            vista = construir_vista(env, j)
            obs = observacion_desde_vista(vista)
            accion = int(heur(vista))                       # el heurístico juega con cualquier nº

            # --- demos para behaviour cloning ---
            obs_list.append(obs)
            acciones.append(accion)
            masks.append(np.asarray(vista.mask, dtype=bool))

            # --- transición pendiente del mismo asiento -> buffer ---
            if j in pendientes:
                po, pa, pr = pendientes.pop(j)
                replay_buffer.push(po, pa, pr, obs, False, vista.mask)

            _, reward, done, _ = env.step(accion)

            if not done:
                pendientes[j] = (obs, accion, reward)
            else:
                finales = _recompensas_finales(env, config['RECOMPENSA_FINAL'])
                replay_buffer.push(obs, accion, reward + finales[j], obs, True, vacia)
                for s, (po, pa, pr) in pendientes.items():    # el resto de asientos cierra su partida
                    replay_buffer.push(po, pa, pr + finales[s], po, True, vacia)

        if ep % 250 == 0:
            print(f"[imit] {ep}/{config['IMIT_EPISODIOS']} partidas de {n} jugadores, "
                  f"{len(acciones)} decisiones, buffer {len(replay_buffer)}")

    # demos agrupadas por clave de observación (tablero_yo, tablero_izq, ..., escalares)
    claves = obs_list[0].keys()
    demos = {k: np.stack([o[k] for o in obs_list]).astype(np.float32) for k in claves}
    return demos, np.array(acciones, dtype=np.int64), np.stack(masks)


def _entrenar_bc(policy_net, demos, acciones, masks, config, device):
    """Cross-entropy sobre las acciones del heurístico (ilegales enmascaradas)."""
    n = len(acciones)
    idx = np.arange(n)
    opt = torch.optim.Adam(policy_net.parameters(), lr=config['IMIT_LR'])
    acc_final = 0.0

    for epoch in range(1, config['IMIT_EPOCHS'] + 1):
        np.random.shuffle(idx)
        tot_loss, aciertos = 0.0, 0
        policy_net.train()
        for i in range(0, n, config['IMIT_BATCH']):
            b = idx[i:i + config['IMIT_BATCH']]
            obs = {k: torch.from_numpy(v[b]).to(device) for k, v in demos.items()}
            a = torch.from_numpy(acciones[b]).to(device)
            m = torch.from_numpy(masks[b]).to(device)

            q = policy_net(obs)
            q = q.masked_fill(~m, -1e9)
            loss = F.cross_entropy(q, a)

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy_net.parameters(), max_norm=10.0)
            opt.step()

            tot_loss += loss.item() * len(b)
            aciertos += int((q.argmax(dim=1) == a).sum().item())

        acc_final = aciertos / n
        print(f"[BC] epoch {epoch:>3}/{config['IMIT_EPOCHS']} | "
              f"loss {tot_loss / n:.4f} | acc {acc_final:.3f}")

    return acc_final

    
def preentrenar_imitacion(policy_net, replay_buffer, config, device):
    """Fase completa de imitación. Devuelve (exactitud BC, FuenteDemos o None)."""
    env = SkyjoEnv(num_jugadores=int(config['JUGADORES']))
    demos, acciones, masks = _partidas_heuristicas(env, replay_buffer, config)
    acc = _entrenar_bc(policy_net, demos, acciones, masks, config, device)
    fuente = FuenteDemos(demos, acciones, masks) if config.get('DEMO_LAMBDA', 0) > 0 else None
    return acc, fuente
        
