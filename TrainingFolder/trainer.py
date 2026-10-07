import sys
import os
import torch
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

# Permitir importar módulos desde la raíz del proyecto
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from skyjo_env import SkyjoEnv
from skyjo_model import SkyjoDQN
from TrainingFolder.replay_buffer import ReplayBuffer
from TrainingFolder.utils import seleccionar_accion

def entrenar_step(policy_net, target_net, replay_buffer, optimizer, batch_size, gamma, device):
    """Paso de optimización usando Double DQN."""
    if len(replay_buffer) < batch_size:
        return

    # 1. Muestrear lote del Replay Buffer
    obs, actions, rewards, next_obs, dones, next_masks = replay_buffer.sample(batch_size)

    # 2. Q(s, a) estimados por la red principal
    q_values = policy_net(obs)
    q_value = q_values.gather(1, actions)  # Shape: [B, 1]

    with torch.no_grad():
        # --- DOUBLE DQN ---
        # A) Seleccionar la mejor acción futura en S' usando la POLICY_NET
        next_q_policy = policy_net(next_obs)
        next_q_policy[~next_masks] = -1e9  # Enmascarar acciones ilegales en S'
        best_actions = torch.argmax(next_q_policy, dim=1, keepdim=True)

        # B) Evaluar esa acción seleccionada usando la TARGET_NET
        next_q_target = target_net(next_obs)
        next_q_value = next_q_target.gather(1, best_actions)
        next_q_value[dones] = 0.0  # Si es estado terminal, Q = 0

        # C) Objetivo Bellman
        expected_q_value = rewards + gamma * next_q_value

    # 3. Pérdida y actualización de pesos (Huber Loss)
    loss = F.smooth_l1_loss(q_value, expected_q_value)
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(policy_net.parameters(), max_norm=10.0)
    optimizer.step()

def entrenar_skyjo(config):
    """Bucle principal de entrenamiento."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Utilizando dispositivo: {device}")

    # --------------------------------------------------------------------------
    # 1. Inicialización del Entorno, Redes y Buffer
    # --------------------------------------------------------------------------
    env = SkyjoEnv(num_jugadores=2)

    # Red Principal (Q-Network) y Red Objetivo (Target Network)
    policy_net = SkyjoDQN(num_jugadores=2).to(device)
    target_net = SkyjoDQN(num_jugadores=2).to(device)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    optimizer = optim.Adam(policy_net.parameters(), lr=config['LR'])

    replay_buffer = ReplayBuffer(capacity=config['BUFFER_CAPACITY'], num_jugadores=2, device=device)

    # --------------------------------------------------------------------------
    # 2. Bucle Principal de Partidas (Episodes)
    # --------------------------------------------------------------------------
    epsilon = config['EPSILON_START']
    pasos_para_decay = 0.80 * config['NUM_EPISODIOS']
    epsilon_decay = (config['EPSILON_END'] / config['EPSILON_START']) ** (1.0 / pasos_para_decay)

    global_step = 0
    transiciones_pendientes = {}

    for episodio in range(1, config['NUM_EPISODIOS'] + 1):
        obs = env.reset()
        done = False
        recompensa_acumulada = 0.0
        transiciones_pendientes.clear()

        while not done:
            jugador_actual = env.turno_actual
            mask = env.get_action_mask(jugador_actual)

            # 1. Seleccionar acción (con la obs rotada del jugador de turno)
            accion = seleccionar_accion(policy_net, obs, env, epsilon, device=device)

            # 2. Si este jugador tenía una jugada pendiente del turno anterior,
            #    su 'next_obs' ES la observación de AHORA (un turno completo después).
            if jugador_actual in transiciones_pendientes:
                prev_obs, prev_accion, prev_reward = transiciones_pendientes[jugador_actual]
                siguiente_mask = env.get_action_mask(jugador_actual)

                replay_buffer.push(
                    obs=prev_obs,
                    action=prev_accion,
                    reward=prev_reward,
                    next_obs=obs,
                    done=False,
                    next_mask=siguiente_mask  # Mascara añadida para el paso intermedio
                )

            # 3. Avanzar el entorno
            siguiente_obs, reward, done, _ = env.step(accion)
            global_step += 1

            if jugador_actual == 0:
                recompensa_acumulada += reward

            # 4. Gestión del almacenamiento según si la partida terminó o no
            if not done:
                # Guardar la jugada en espera para este jugador hasta que llegue su próximo turno
                transiciones_pendientes[jugador_actual] = (obs, accion, reward)
            else:
                # ==================================================================
                # GAME OVER: CALCULAR RECOMPENSAS FINALES OFICIALES
                # ==================================================================
                # Obtener puntos reales procesando tableros finales y regla del cerrador
                puntos = env.calcular_puntuaciones_finales()
                puntos_j0, puntos_j1 = float(puntos[0]), float(puntos[1])

                # Asignar premio/castigo final según el ganador (menos puntos = mejor)
                if puntos_j0 < puntos_j1:
                    reward_final_j0 =  (puntos_j1 - puntos_j0) * 1.5
                    reward_final_j1 = -(puntos_j1 - puntos_j0) * 1.5
                elif puntos_j1 < puntos_j0:
                    reward_final_j1 =  (puntos_j0 - puntos_j1) * 1.5
                    reward_final_j0 = -(puntos_j0 - puntos_j1) * 1.5
                else:
                    reward_final_j0 = 0.0
                    reward_final_j1 = 0.0

                # ------------------------------------------------------------------
                # A) Cerrar la última jugada del jugador actual (el que acaba de actuar)
                # ------------------------------------------------------------------
                rf_actual = reward_final_j0 if jugador_actual == 0 else reward_final_j1
                replay_buffer.push(
                    obs=obs,
                    action=accion,
                    reward=reward + rf_actual,
                    next_obs=siguiente_obs,
                    done=True,
                    next_mask=np.zeros(15, dtype=bool) # No hay acciones futuras posibles
                )

                # ------------------------------------------------------------------
                # B) Cerrar la jugada que le quedaba pendiente al RIVAL
                # ------------------------------------------------------------------
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
                        next_mask=np.zeros(15, dtype=bool)
                    )

            # Avanzar estado para el próximo turno del loop
            obs = siguiente_obs

            # 5. Optimización / Entrenamiento de la red
            if len(replay_buffer) > config['WARMUP_STEPS'] and global_step % config['TRAIN_FREQ'] == 0:
                entrenar_step(
                    policy_net=policy_net,
                    target_net=target_net,
                    replay_buffer=replay_buffer,
                    optimizer=optimizer,
                    batch_size=config['BATCH_SIZE'],
                    gamma=config['GAMMA'],
                    device=device
                )

            # 6. Actualizar la red Target periódicamente
            if global_step % config['TARGET_UPDATE_FREQ'] == 0:
                target_net.load_state_dict(policy_net.state_dict())

        # Decaimiento de Epsilon al final de cada episodio
        epsilon = max(config['EPSILON_END'], epsilon * epsilon_decay)

        # Print de control cada 100 episodios
        if episodio % 100 == 0:
            print(f"Episodio: {episodio:4d} | Recompensa J0: {recompensa_acumulada:6.1f} | Epsilon: {epsilon:.3f} | Pasos: {global_step}")

    # Guardar el modelo entrenado
    torch.save(policy_net.state_dict(), config['SAVE_PATH'])
    print(f"\n¡Entrenamiento completado! Modelo guardado en '{config['SAVE_PATH']}'.")
