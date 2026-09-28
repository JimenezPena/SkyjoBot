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

    for episodio in range(1, config['NUM_EPISODIOS'] + 1):
        obs = env.reset()
        done = False
        recompensa_acumulada = 0.0
        transiciones_pendientes.clear()

        while not done:
            jugador_actual = env.turno_actual
            accion = seleccionar_accion(policy_net, obs, env, epsilon, device=device)

            if jugador_actual in transiciones_pendientes:
                prev_obs, prev_accion, prev_reward = transiciones_pendientes[jugador_actual]
                siguiente_mask = env.get_action_mask(jugador_actual)
                replay_buffer.push(prev_obs, prev_accion, prev_reward, obs, False, siguiente_mask)

            siguiente_obs, reward, done, _ = env.step(accion)
            global_step += 1

            if jugador_actual == 0:
                recompensa_acumulada += reward

            if not done:
                transiciones_pendientes[jugador_actual] = (obs, accion, reward)
            else:
                puntos = env.calcular_puntuaciones_finales()
                puntos_j0, puntos_j1 = float(puntos[0]), float(puntos[1])

                if puntos_j0 < puntos_j1:
                    rf_j0, rf_j1 = 40.0, -(puntos_j1 - puntos_j0) * 1.5
                elif puntos_j1 < puntos_j0:
                    rf_j0, rf_j1 = -(puntos_j0 - puntos_j1) * 1.5, 40.0
                else:
                    rf_j0, rf_j1 = 0.0, 0.0

                rf_actual = rf_j0 if jugador_actual == 0 else rf_j1
                replay_buffer.push(obs, accion, reward + rf_actual, siguiente_obs, True, np.zeros(15, dtype=bool))

                rival_id = (jugador_actual + 1) % env.num_jugadores
                if rival_id in transiciones_pendientes:
                    prev_obs_r, prev_accion_r, prev_reward_r = transiciones_pendientes[rival_id]
                    rf_rival = rf_j0 if rival_id == 0 else rf_j1
                    replay_buffer.push(prev_obs_r, prev_accion_r, prev_reward_r + rf_rival, siguiente_obs, True, np.zeros(15, dtype=bool))

            obs = siguiente_obs

            if len(replay_buffer) > config['WARMUP_STEPS'] and global_step % config['TRAIN_FREQ'] == 0:
                entrenar_step(policy_net, target_net, replay_buffer, optimizer, config['BATCH_SIZE'], config['GAMMA'], device)

            if global_step % config['TARGET_UPDATE_FREQ'] == 0:
                target_net.load_state_dict(policy_net.state_dict())

        epsilon = max(config['EPSILON_END'], epsilon * epsilon_decay)

        if episodio % 100 == 0:
            print(f"Episodio: {episodio:4d} | Recompensa J0: {recompensa_acumulada:6.1f} | Epsilon: {epsilon:.3f} | Pasos: {global_step}")

    torch.save(policy_net.state_dict(), config['SAVE_PATH'])
    print(f"\n¡Entrenamiento completado! Modelo guardado en '{config['SAVE_PATH']}'.")
