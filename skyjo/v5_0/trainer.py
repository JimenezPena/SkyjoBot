"""Entrenamiento self-play de v5.0: una sola red compartida juega todos los asientos de una partida de N jugadores
y cada asiento aporta sus propias transiciones (observación relativa a su asiento). Recompensa:
  - intermedia: la del entorno (shaping por potencial, igual que 4.2)
  - final: RECOMPENSA_FINAL * (media de puntos de los rivales - mis puntos)   [con 2 jugadores = 4.2]"""
import os
import random
import time

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

from skyjo.vista import construir_vista
from .agents import evaluar_vs_heuristicos, imprimir_vs_heuristicos
from .buffer import ReplayBuffer
from .env import SkyjoEnv, N_ACCIONES
from .model import SkyjoDQN
from .observacion import observacion_desde_vista, a_tensores
from .imitacion import preentrenar_imitacion, paso_margen

def guardar_checkpoint(policy_net, target_net, optimizer, global_step, episodio, mejor_wr, filepath="/content/drive/MyDrive/SkyjoBot/checkpoint.pt"):
    # Asegúrate de que el directorio existe en Google Drive
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    checkpoint = {
        'policy_net_state': policy_net.state_dict(),
        'target_net_state': target_net.state_dict(),
        'optimizer_state': optimizer.state_dict(),
        'global_step': global_step,
        'episodio': episodio,
        'mejor_wr': mejor_wr
    }
    torch.save(checkpoint, filepath)
    print(f"[Checkpoint] Guardado correctamente en episodio {episodio} (step {global_step})")
    
def cargar_checkpoint(filepath, policy_net, target_net, optimizer, device):
    if os.path.exists(filepath):
        print(f"[Checkpoint] Cargando punto de control desde {filepath}...")
        checkpoint = torch.load(filepath, map_location=device)
        
        policy_net.load_state_dict(checkpoint['policy_net_state'])
        target_net.load_state_dict(checkpoint['target_net_state'])
        optimizer.load_state_dict(checkpoint['optimizer_state'])
        
        global_step = checkpoint['global_step']
        episodio_inicio = checkpoint['episodio'] + 1
        mejor_wr = checkpoint.get('mejor_wr', -1.0)
        
        print(f"[Checkpoint] Reanudando desde episodio {episodio_inicio}, global_step {global_step}, mejor_wr: {100 * mejor_wr:.1f}%")
        return global_step, episodio_inicio, mejor_wr
    else:
        print("[Checkpoint] No se encontró ningún checkpoint previo. Empezando desde cero.")
        return 0, 1, -1.0
            
def entrenar_step(policy_net, target_net, replay_buffer, optimizer, batch_size, gamma, device):
    """Paso de optimización con Double Dueling DQN."""
    if len(replay_buffer) < batch_size:
        return None
    obs, actions, rewards, next_obs, dones, next_masks = replay_buffer.sample(batch_size)
    q_value = policy_net(obs).gather(1, actions)
    with torch.no_grad():
        next_q_policy = policy_net(next_obs)
        next_q_policy[~next_masks] = -1e9
        best = torch.argmax(next_q_policy, dim=1, keepdim=True)
        next_q = target_net(next_obs).gather(1, best)
        next_q[dones] = 0.0
        expected = rewards + gamma * next_q
    loss = F.smooth_l1_loss(q_value, expected)
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(policy_net.parameters(), max_norm=10.0)
    optimizer.step()
    return loss.item()


def _greedy(net, obs, mask, device):
    net.eval()
    with torch.no_grad():
        q = net(a_tensores([obs], device))
    q[0, ~torch.tensor(mask, device=device)] = -1e9
    return int(q.argmax(dim=1).item())


def entrenar_skyjo(config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n = int(config['JUGADORES'])
    print(f"Utilizando dispositivo: {device} | jugadores por partida: {n}")

    env = SkyjoEnv(num_jugadores=n)
    policy_net, target_net = SkyjoDQN().to(device), SkyjoDQN().to(device)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()
    optimizer = optim.Adam(policy_net.parameters(), lr=config['LR'])

    env.reset()
    buffer = ReplayBuffer(config['BUFFER_CAPACITY'], observacion_desde_vista(construir_vista(env)), device)
    vacia = np.zeros(N_ACCIONES, dtype=bool)

    n_ep = config['NUM_EPISODIOS']
    epsilon = config['EPSILON_START']
    fuente = None
    
    os.makedirs(config['CKPT_DIR'], exist_ok=True)

    decay = (config['EPSILON_END'] / config['EPSILON_START']) ** (1.0 / (0.80 * n_ep))
    hist_pts, hist_loss, t0 = [], [], time.time()
    eval_n = list(config.get('EVAL_JUGADORES', [n]))
    
    # ---- Cargando checkpoint guardado si existe. Comenzar nuevo en caso contrario.
    global_step, inicio_episodio, mejor_wr = cargar_checkpoint(f"{config['CKPT_DIR']}/checkpoint.pt", policy_net, target_net, optimizer, device)
        
    # ---- Fase previa: imitar al heurístico (behaviour cloning + buffer de demos) ----
    if inicio_episodio == 1 and config.get('IMIT_EPISODIOS', 0) > 0:
        acc, fuente = preentrenar_imitacion(policy_net, buffer, config, device)
        target_net.load_state_dict(policy_net.state_dict())
        target_net.eval()
        epsilon = config.get('EPSILON_START_IMIT', 0.3)
        
        print(f"Imitación completada (acc {acc:.3f}). "
              f"DQN desde epsilon={epsilon:.2f}, buffer={len(buffer)}")
        res = evaluar_vs_heuristicos(policy_net, device, n, config['EVAL_GAMES'])
        imprimir_vs_heuristicos(f"[Tras imitación: 1 DQN vs {n - 1} heurísticos", res)          
    
        # calibrar la escala de Q con TD sobre las demos (política protegida por el margen)
        for _ in range(config.get('CALIBRATION_PRESTEPS', 3000)):
            policy_net.train()
            
            # 1. Actualización DQN estándar
            entrenar_step(policy_net, target_net, buffer, optimizer, config['BATCH_SIZE'],
                          config['GAMMA'], device)
            
            # 2. Mantener la protección del margen DQfD para no perder la imitación
            if fuente is not None:
                paso_margen(policy_net, fuente, optimizer, config, device)

            if _ % config['TARGET_UPDATE_FREQ'] == 0:
                target_net.load_state_dict(policy_net.state_dict())
        
        print(f"Recalibración de la escala Q completada (acc {acc:.3f}). "
              f"DQN desde epsilon={epsilon:.2f}, buffer={len(buffer)}")
        res = evaluar_vs_heuristicos(policy_net, device, n, config['EVAL_GAMES'])
        imprimir_vs_heuristicos(f"[Tras imitación: 1 DQN vs {n - 1} heurísticos", res)              
                  
    
    for episodio in range(inicio_episodio, n_ep + 1):
        env.reset()
        done, pendientes, rec_j0 = False, {}, 0.0
        while not done:
            j = env.turno_actual
            vista = construir_vista(env, j)
            obs = observacion_desde_vista(vista)
            if random.random() < epsilon:
                accion = int(random.choice(np.flatnonzero(vista.mask)))
            else:
                accion = _greedy(policy_net, obs, vista.mask, device)

            if j in pendientes:
                po, pa, pr = pendientes[j]
                buffer.push(po, pa, pr, obs, False, vista.mask)

            _, reward, done, _ = env.step(accion)
            global_step += 1
            if j == 0:
                rec_j0 += reward

            if not done:
                pendientes[j] = (obs, accion, reward)
            else:
                puntos = np.asarray(env.calcular_puntuaciones_finales(), dtype=np.float64)
                media_riv = (puntos.sum() - puntos) / (n - 1)
                finales = config['RECOMPENSA_FINAL'] * (media_riv - puntos)
                hist_pts.append(puntos)
                buffer.push(obs, accion, reward + finales[j], obs, True, vacia)
                for s, (po, pa, pr) in pendientes.items():
                    if s != j:
                        buffer.push(po, pa, pr + finales[s], po, True, vacia)

            if len(buffer) > config['WARMUP_STEPS'] and global_step % config['TRAIN_FREQ'] == 0:
                policy_net.train()
                
                # 1. Entrenamiento normal de DQN
                loss = entrenar_step(policy_net, target_net, buffer, optimizer, config['BATCH_SIZE'],
                                     config['GAMMA'], device)
                if loss is not None:
                    hist_loss.append(loss)
                
                # 2. Paso de margen DQfD (si estamos dentro del límite de demos)
                if fuente is not None and global_step < config['DEMO_STEPS']:
                    paso_margen(policy_net, fuente, optimizer, config, device)
        
            if global_step % config['TARGET_UPDATE_FREQ'] == 0:
                target_net.load_state_dict(policy_net.state_dict())

        epsilon = max(config['EPSILON_END'], epsilon * decay)

        if episodio % config['LOG_EVERY'] == 0:
            p = np.mean(hist_pts[-config['LOG_EVERY']:], axis=0)
            l = np.mean(hist_loss[-500:]) if hist_loss else float('nan')
            print(f"Ep {episodio:>6}/{n_ep} | eps {epsilon:.3f} | pasos {global_step:>8} | "
                  f"pts self-play {' / '.join(f'{x:5.1f}' for x in p)} | loss {l:.4f} | "
                  f"Recompensa J0: {rec_j0:6.1f} | {(time.time() - t0) / 60:.1f} min")

        if episodio % config['EVAL_EVERY'] == 0:
            for k, nj in enumerate(eval_n):
                res = evaluar_vs_heuristicos(policy_net, device, nj, config['EVAL_GAMES'])
                imprimir_vs_heuristicos(f"[ep {episodio}] 1 DQN vs {nj - 1} heurísticos", res)
                if k == 0:
                    wr = res['_dqn']['winrate']
            torch.save(policy_net.state_dict(), f"{config['CKPT_DIR']}/ep{episodio}.pth")
            if wr > mejor_wr:
                mejor_wr = wr
                torch.save(policy_net.state_dict(), f"{config['CKPT_DIR']}/best.pth")
                print(f"   -> nuevo mejor checkpoint (winrate {100 * mejor_wr:.1f}% con {eval_n[0]} jugadores)")
                guardar_checkpoint(policy_net, target_net, optimizer, global_step, episodio, mejor_wr,  f"{config['CKPT_DIR']}/best_checkpoint.pt")

        if episodio % config['SAVE_EVERY'] == 0:
            guardar_checkpoint(policy_net, target_net, optimizer, global_step, episodio, mejor_wr,  f"{config['CKPT_DIR']}/checkpoint.pt")

    torch.save(policy_net.state_dict(), config['SAVE_PATH'])
    print(f"\n¡Entrenamiento completado! Modelo guardado en '{config['SAVE_PATH']}'. Mejor winrate: {100 * mejor_wr:.1f}%")
