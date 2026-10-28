"""Entrenamiento de v5.0 (self-play o contra una mezcla de rivales, ver rivales.py): una sola red compartida de N jugadores
y cada asiento aporta sus propias transiciones (observación relativa a su asiento). Recompensa:
  - intermedia: la del entorno (shaping por potencial, igual que 4.2)
  - final: RECOMPENSA_FINAL * (media de puntos de los rivales - mis puntos)   [con 2 jugadores = 4.2]
    + RECOMPENSA_VICTORIA * (victoria - 1/n)   (0 = desactivado)"""
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
from .imitacion import preentrenar_imitacion, paso_margen, recompensas_finales, asientos_aprendiz
from .rivales import PoolRivales, pesos_en

def guardar_checkpoint(policy_net, target_net, optimizer, global_step, episodio, mejor_pts, epsilon, buffer, filepath="/content/drive/MyDrive/SkyjoBot/checkpoint.pt"):
    # Asegúrate de que el directorio existe en Google Drive
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    checkpoint = {
        'policy_net_state': policy_net.state_dict(),
        'target_net_state': target_net.state_dict(),
        'optimizer_state': optimizer.state_dict(),
        'global_step': global_step,
        'episodio': episodio,
        'mejor_pts': mejor_pts,          # menor puntuación media del DQN vs heurísticos (los checkpoints antiguos guardaban 'mejor_wr')
        'epsilon': epsilon,
        'buffer_state': {
            'ptr': buffer.ptr,
            'size': buffer.size,
            'obs': buffer.obs,
            'next_obs': buffer.next_obs,
            'actions': buffer.actions,
            'rewards': buffer.rewards,
            'dones': buffer.dones,
            'next_masks': buffer.next_masks
        }
    }
    torch.save(checkpoint, filepath)
    print(f"[Checkpoint] Guardado correctamente en episodio {episodio} (step {global_step})")
    
def cargar_checkpoint(filepath, policy_net, target_net, optimizer, buffer, device):
    if os.path.exists(filepath):
        print(f"[Checkpoint] Cargando punto de control desde {filepath}...")
        checkpoint = torch.load(filepath, map_location=device, weights_only=False)   # el buffer son arrays numpy (torch>=2.6 lo rechaza por defecto)
        
        policy_net.load_state_dict(checkpoint['policy_net_state'])
        target_net.load_state_dict(checkpoint['target_net_state'])
        optimizer.load_state_dict(checkpoint['optimizer_state'])
        
        global_step = checkpoint['global_step']
        episodio_inicio = checkpoint['episodio'] + 1
        mejor_pts = checkpoint.get('mejor_pts', float('inf'))      # checkpoint antiguo (winrate): la 1ª evaluación fija el mejor
        epsilon = checkpoint.get('epsilon', None)
        
        # Restaurar el buffer si existe en el checkpoint
        if 'buffer_state' in checkpoint:
            bs = checkpoint['buffer_state']
            buffer.ptr = bs['ptr']
            buffer.size = bs['size']
            buffer.obs = bs['obs']
            buffer.next_obs = bs['next_obs']
            buffer.actions = bs['actions']
            buffer.rewards = bs['rewards']
            buffer.dones = bs['dones']
            buffer.next_masks = bs['next_masks']
            print(f"[Checkpoint] Buffer restaurado con éxito ({buffer.size} elementos).")
            
        print(f"[Checkpoint] Reanudando desde episodio {episodio_inicio}, global_step {global_step}, mejor_pts: {mejor_pts:.2f}, epsilon: {epsilon}")
        return global_step, episodio_inicio, mejor_pts, epsilon
    else:
        print("[Checkpoint] No se encontró ningún checkpoint previo. Empezando desde cero.")
        return 0, 1, float('inf'), None
            
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
    fuente = None
    
    os.makedirs(config['CKPT_DIR'], exist_ok=True)

    decay = (config['EPSILON_END'] / config['EPSILON_START']) ** (1.0 / (0.80 * n_ep))
    hist_pts, hist_apr, hist_loss, t0 = [], [], [], time.time()
    eval_n = [n] + [x for x in config.get('EVAL_JUGADORES', []) if x != n]     # el entrenado va primero: decide el mejor checkpoint
    
    # ---- Cargando checkpoint guardado si existe. Comenzar nuevo en caso contrario.
    global_step, inicio_episodio, mejor_pts, ckpt_epsilon = cargar_checkpoint(f"{config['CKPT_DIR']}/checkpoint.pt", policy_net, target_net, optimizer, buffer, device)
    
    epsilon = ckpt_epsilon if ckpt_epsilon is not None else config['EPSILON_START']

    # ---- Rivales: mezcla de heurísticos / red actual / checkpoints (RIVALES=None -> self-play puro, todos aprenden)
    pool, pool_path = None, f"{config['CKPT_DIR']}/pool.pt"
    if config.get('RIVALES'):
        pool = PoolRivales(policy_net, device, config.get('RIVAL_EPSILON', 0.05), config.get('POOL_MAX', 5),
                           config.get('RIVALES_CKPT', []))
        if os.path.exists(pool_path):
            pool.cargar(pool_path)
        print(f"Rivales: {config['RIVALES']} -> {config.get('RIVALES_FIN') or 'sin cambio'} | "
              f"{config.get('ASIENTOS_APRENDIZ', 1)} asiento(s) aprendiz al azar | checkpoints en el pool: {len(pool)}")
        
    # ---- Fase previa: imitar al heurístico (behaviour cloning + buffer de demos) ----
    if inicio_episodio == 1 and config.get('IMIT_EPISODIOS', 0) > 0:
        acc, fuente = preentrenar_imitacion(policy_net, buffer, config, device)
        target_net.load_state_dict(policy_net.state_dict())
        target_net.eval()
        epsilon = config.get('EPSILON_START_IMIT', 0.3)
        
        print(f"Imitación completada (acc {acc:.3f}). "
              f"DQN desde epsilon={epsilon:.2f}, buffer={len(buffer)}")
        res = evaluar_vs_heuristicos(policy_net, device, n, config['EVAL_GAMES'], semilla=config.get('EVAL_SEMILLA'))
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
        res = evaluar_vs_heuristicos(policy_net, device, n, config['EVAL_GAMES'], semilla=config.get('EVAL_SEMILLA'))
        imprimir_vs_heuristicos(f"[Tras imitación: 1 DQN vs {n - 1} heurísticos", res)              
        if pool is not None:
            pool.anadir_instantanea(policy_net, 0)       # la red imitadora es el primer 'checkpoint antiguo'
    
    # Barajas simétricas (solo con rivales): n episodios seguidos comparten baraja y el aprendiz rota por los n asientos,
    # así que ve cada mano inicial de cada baraja. Con self-play puro (RIVALES=None) todos los asientos ya aprenden de
    # cada baraja, y no hace falta.
    simetrico = pool is not None and bool(config.get('BARAJAS_SIMETRICAS', False))
    semilla_baraja = None
    if simetrico:
        print(f"Barajas simétricas: cada baraja se juega {n} veces con el aprendiz rotando por los asientos")
    print(f"Recompensa final: referencia = {config.get('RECOMPENSA_REF', 'media')} de los rivales")

    for episodio in range(inicio_episodio, n_ep + 1):
        if simetrico:
            if semilla_baraja is None or (episodio - 1) % n == 0:
                semilla_baraja = random.randrange(2 ** 31)
            env = SkyjoEnv(num_jugadores=n, semilla=semilla_baraja)         # misma semilla = misma baraja y manos iniciales
        env.reset()
        if pool is not None:
            k = min(max(int(config.get('ASIENTOS_APRENDIZ', 1)), 1), n)
            aprendices = asientos_aprendiz(episodio, n, k) if simetrico else set(random.sample(range(n), k))
            pesos = pesos_en(config, episodio / n_ep)
            rivales = {s: pool.elegir(pesos) for s in range(n) if s not in aprendices}
        else:
            aprendices, rivales = set(range(n)), {}
        done, pendientes, rec_apr = False, {}, 0.0
        while not done:
            j = env.turno_actual
            vista = construir_vista(env, j)

            if j in rivales:                                                 # turno de un rival: no genera datos
                accion = int(rivales[j](vista))
                if not vista.mask[accion]:
                    accion = int(random.choice(np.flatnonzero(vista.mask)))
                _, reward, done, _ = env.step(accion)
            else:
                obs = observacion_desde_vista(vista)
                if random.random() < epsilon:
                    accion = int(random.choice(np.flatnonzero(vista.mask)))
                else:
                    accion = _greedy(policy_net, obs, vista.mask, device)

                if j in pendientes:
                    po, pa, pr = pendientes[j]
                    buffer.push(po, pa, pr, obs, False, vista.mask)

                _, reward, done, _ = env.step(accion)
                global_step += 1                                             # solo cuentan los pasos del aprendiz
                rec_apr += reward
                if not done:
                    pendientes[j] = (obs, accion, reward)

            if done:
                puntos = np.asarray(env.calcular_puntuaciones_finales(), dtype=np.float64)
                finales = recompensas_finales(puntos, config)
                hist_pts.append(puntos)
                hist_apr.append((puntos[sorted(aprendices)].mean(),
                                 puntos[[s for s in range(n) if s not in aprendices]].mean() if rivales else float('nan')))
                if j not in rivales:
                    buffer.push(obs, accion, reward + finales[j], obs, True, vacia)
                for s, (po, pa, pr) in pendientes.items():
                    if s != j:
                        buffer.push(po, pa, pr + finales[s], po, True, vacia)

            if j in rivales:
                continue                                                     # entrenar solo tras pasos del aprendiz

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

        if pool is not None and episodio % config.get('POOL_EVERY', 1000) == 0:
            pool.anadir_instantanea(policy_net, episodio)

        if episodio % config['LOG_EVERY'] == 0:
            l = np.mean(hist_loss[-500:]) if hist_loss else float('nan')
            if pool is not None:
                pa = np.nanmean(np.array(hist_apr[-config['LOG_EVERY']:], dtype=float), axis=0)
                pts = f"pts aprendiz {pa[0]:5.1f} | rivales {pa[1]:5.1f}"
            else:
                pts = "pts self-play " + ' / '.join(f'{x:5.1f}' for x in np.mean(hist_pts[-config['LOG_EVERY']:], axis=0))
            print(f"Ep {episodio:>6}/{n_ep} | eps {epsilon:.3f} | pasos {global_step:>8} | {pts} | loss {l:.4f} | "
                  f"Recompensa aprendiz: {rec_apr:6.1f} | {(time.time() - t0) / 60:.1f} min")

        if episodio % config['EVAL_EVERY'] == 0:
            for k, nj in enumerate(eval_n):
                res = evaluar_vs_heuristicos(policy_net, device, nj, config['EVAL_GAMES'], semilla=config.get('EVAL_SEMILLA'))
                imprimir_vs_heuristicos(f"[ep {episodio}] 1 DQN vs {nj - 1} heurísticos", res)
                if nj == n:
                    pts = res['_dqn']['puntos']
            torch.save(policy_net.state_dict(), f"{config['CKPT_DIR']}/ep{episodio}.pth")
            if pts < mejor_pts:                                              # el mejor checkpoint = menos puntos medios
                mejor_pts = pts
                torch.save(policy_net.state_dict(), f"{config['CKPT_DIR']}/best.pth")
                print(f"   -> nuevo mejor checkpoint (puntos medios {mejor_pts:.2f} con {n} jugadores)")
                
        if episodio % config['SAVE_EVERY'] == 0:
            guardar_checkpoint(policy_net, target_net, optimizer, global_step, episodio, mejor_pts, epsilon, buffer, f"{config['CKPT_DIR']}/checkpoint.pt")
            if pool is not None:
                pool.guardar(pool_path)

    torch.save(policy_net.state_dict(), config['SAVE_PATH'])
    print(f"\n¡Entrenamiento completado! Modelo guardado en '{config['SAVE_PATH']}'. Mejores puntos medios vs heurísticos: {mejor_pts:.2f}")
