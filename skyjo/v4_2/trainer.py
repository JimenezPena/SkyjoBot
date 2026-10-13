from .env import SkyjoEnv, N_ACCIONES
from .model import SkyjoDQN
from .buffer import ReplayBuffer
from .acciones import seleccionar_accion
from .agents import duelo, imprimir_duelo, agente_dqn, agente_heuristico

# ==============================================================================
# TrainingFolder/trainer.py  (v4.2: Double DQN + Dueling + evaluación periódica vs heurístico)
# ==============================================================================
import os
import time
import torch
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

def entrenar_step(policy_net, target_net, replay_buffer, optimizer, batch_size, gamma, device):
    """Paso de optimización usando Double Dueling DQN."""
    if len(replay_buffer) < batch_size:
        return None

    obs, actions, rewards, next_obs, dones, next_masks = replay_buffer.sample(batch_size)

    q_values = policy_net(obs)
    q_value = q_values.gather(1, actions)

    with torch.no_grad():
        next_q_policy = policy_net(next_obs)
        next_q_policy[~next_masks] = -1e9
        best_actions = torch.argmax(next_q_policy, dim=1, keepdim=True)

        next_q_target = target_net(next_obs)
        next_q_value = next_q_target.gather(1, best_actions)
        next_q_value[dones] = 0.0

        expected_q_value = rewards + gamma * next_q_value

    loss = F.smooth_l1_loss(q_value, expected_q_value)
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(policy_net.parameters(), max_norm=10.0)
    optimizer.step()
    return loss.item()

def entrenar_skyjo(config):
    """Bucle principal de entrenamiento."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Utilizando dispositivo: {device}")

    env = SkyjoEnv(num_jugadores=2)

    policy_net = SkyjoDQN(num_jugadores=2).to(device)
    target_net = SkyjoDQN(num_jugadores=2).to(device)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    optimizer = optim.Adam(policy_net.parameters(), lr=config['LR'])
    replay_buffer = ReplayBuffer(capacity=config['BUFFER_CAPACITY'], num_jugadores=2, device=device)

    epsilon = config['EPSILON_START']
    pasos_para_decay = 0.80 * config['NUM_EPISODIOS']
    epsilon_decay = (config['EPSILON_END'] / config['EPSILON_START']) ** (1.0 / pasos_para_decay)

    global_step = 0
    transiciones_pendientes = {}
    os.makedirs(config['CKPT_DIR'], exist_ok=True)
    n_ep = config['NUM_EPISODIOS']
    mejor_wr, hist_pts, hist_loss = -1.0, [], []
    t0 = time.time()

    for episodio in range(1, config['NUM_EPISODIOS'] + 1):
        obs = env.reset()
        done = False
        recompensa_acumulada = 0.0
        transiciones_pendientes.clear()

        while not done:
            jugador_actual = env.turno_actual
            mask = env.get_action_mask(jugador_actual)

            accion = seleccionar_accion(policy_net, obs, env, epsilon, device=device)

            if jugador_actual in transiciones_pendientes:
                prev_obs, prev_accion, prev_reward = transiciones_pendientes[jugador_actual]
                siguiente_mask = env.get_action_mask(jugador_actual)

                replay_buffer.push(
                    obs=prev_obs,
                    action=prev_accion,
                    reward=prev_reward,
                    next_obs=obs,
                    done=False,
                    next_mask=siguiente_mask
                )

            siguiente_obs, reward, done, _ = env.step(accion)
            global_step += 1

            if jugador_actual == 0:
                recompensa_acumulada += reward

            if not done:
                transiciones_pendientes[jugador_actual] = (obs, accion, reward)
            else:
                puntos = env.calcular_puntuaciones_finales()
                puntos_j0, puntos_j1 = float(puntos[0]), float(puntos[1])
                hist_pts.append(puntos)

                if puntos_j0 < puntos_j1:
                    reward_final_j0 =  (puntos_j1 - puntos_j0) * 1.5
                    reward_final_j1 = -(puntos_j1 - puntos_j0) * 1.5
                elif puntos_j1 < puntos_j0:
                    reward_final_j1 =  (puntos_j0 - puntos_j1) * 1.5
                    reward_final_j0 = -(puntos_j0 - puntos_j1) * 1.5
                else:
                    reward_final_j0 = 0.0
                    reward_final_j1 = 0.0

                rf_actual = reward_final_j0 if jugador_actual == 0 else reward_final_j1
                replay_buffer.push(
                    obs=obs,
                    action=accion,
                    reward=reward + rf_actual,
                    next_obs=siguiente_obs,
                    done=True,
                    next_mask=np.zeros(N_ACCIONES, dtype=bool)
                )

                rival_id = (jugador_actual + 1) % env.num_jugadores
                if rival_id in transiciones_pendientes:
                    prev_obs_r, prev_accion_r, prev_reward_r = transiciones_pendientes[rival_id]
                    rf_rival = reward_final_j0 if rival_id == 0 else reward_final_j1

                    replay_buffer.push(
                        obs=prev_obs_r,
                        action=prev_accion_r,
                        reward=prev_reward_r + rf_rival,
                        next_obs=siguiente_obs,
                        done=True,
                        next_mask=np.zeros(N_ACCIONES, dtype=bool)
                    )

            obs = siguiente_obs

            if len(replay_buffer) > config['WARMUP_STEPS'] and global_step % config['TRAIN_FREQ'] == 0:
                loss = entrenar_step(
                    policy_net=policy_net,
                    target_net=target_net,
                    replay_buffer=replay_buffer,
                    optimizer=optimizer,
                    batch_size=config['BATCH_SIZE'],
                    gamma=config['GAMMA'],
                    device=device
                )
                if loss is not None:
                    hist_loss.append(loss)

            if global_step % config['TARGET_UPDATE_FREQ'] == 0:
                target_net.load_state_dict(policy_net.state_dict())

        epsilon = max(config['EPSILON_END'], epsilon * epsilon_decay)

        if episodio % config['LOG_EVERY'] == 0:
            p = np.mean(hist_pts[-config['LOG_EVERY']:], axis=0)
            l = np.mean(hist_loss[-500:]) if hist_loss else float('nan')
            print(f"Ep {episodio:>6}/{n_ep} | eps {epsilon:.3f} | pasos {global_step:>8} | "
                  f"pts self-play J0/J1 {p[0]:5.1f}/{p[1]:5.1f} | loss {l:.4f} | Recompensa J0: {recompensa_acumulada:6.1f} | {(time.time()-t0)/60:.1f} min")

        # Evaluación periódica frente al bot heurístico (greedy, sin exploración)
        if episodio % config['EVAL_EVERY'] == 0:
            res = duelo(agente_dqn(policy_net, device), agente_heuristico, config['EVAL_GAMES'])
            imprimir_duelo(f"[ep {episodio}] DQN vs heurístico", res)
            torch.save(policy_net.state_dict(), f"{config['CKPT_DIR']}/ep{episodio}.pth")
            if res['winrate'] > mejor_wr:
                mejor_wr = res['winrate']
                torch.save(policy_net.state_dict(), f"{config['CKPT_DIR']}/best.pth")
                print(f"   -> nuevo mejor checkpoint (winrate {100*mejor_wr:.1f}%)")

    torch.save(policy_net.state_dict(), config['SAVE_PATH'])
    print(f"\n¡Entrenamiento completado! Modelo guardado en '{config['SAVE_PATH']}'. Mejor winrate vs heurístico: {100*mejor_wr:.1f}%")
