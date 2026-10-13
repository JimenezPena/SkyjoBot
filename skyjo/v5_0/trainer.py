from .env import SkyjoEnv
from .model import SkyjoDQN
from .buffer import ReplayBuffer, NStepCollector
from .acciones import seleccionar_accion
from .agents import duelo, imprimir_duelo, agente_dqn, agente_heuristico

# ============================================================
# Trainer (v5.0): Double DQN + Dueling enmascarado + n-step + soft update
# ============================================================
import os
import time
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim


def entrenar_step(policy_net, target_net, buffer, optimizer, batch_size):
    obs, mask, actions, rewards, gpow, nobs, dones, nmask = buffer.sample(batch_size)
    q = policy_net(obs, mask).gather(1, actions)
    with torch.no_grad():
        q_pol = policy_net(nobs, nmask).masked_fill(~nmask, -1e9)
        mejor = q_pol.argmax(dim=1, keepdim=True)                       # Double DQN
        q_next = target_net(nobs, nmask).gather(1, mejor).masked_fill(dones, 0.0)
        target = rewards + gpow * q_next                                # gpow = gamma^k del n-step
    loss = F.smooth_l1_loss(q, target)
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(policy_net.parameters(), 10.0)
    optimizer.step()
    return loss.item()


@torch.no_grad()
def actualizar_target_suave(policy_net, target_net, tau):
    for pt, p in zip(target_net.parameters(), policy_net.parameters()):
        pt.mul_(1.0 - tau).add_(p, alpha=tau)


def entrenar_skyjo(config):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Dispositivo: {device}")
    env = SkyjoEnv(num_jugadores=2)
    policy_net = SkyjoDQN(2).to(device)
    target_net = SkyjoDQN(2).to(device)
    target_net.load_state_dict(policy_net.state_dict())
    optimizer = optim.Adam(policy_net.parameters(), lr=config['LR'])
    buffer = ReplayBuffer(config['BUFFER_CAPACITY'], 2, device)
    collector = NStepCollector(buffer, config['N_STEP'], config['GAMMA'])
    os.makedirs(config['CKPT_DIR'], exist_ok=True)

    rs, tc = config['REWARD_SCALE'], config['TERMINAL_COEF']
    n_ep = config['NUM_EPISODIOS']
    ep_decay = config['EPS_DECAY_FRAC'] * n_ep
    epsilon, global_step = config['EPSILON_START'], 0
    mejor_wr, hist_pts, hist_loss = -1.0, [], []
    t0 = time.time()

    for ep in range(1, n_ep + 1):
        epsilon = max(config['EPSILON_END'],
                      config['EPSILON_START'] - (config['EPSILON_START'] - config['EPSILON_END']) * ep / ep_decay)
        obs, done = env.reset(), False
        collector.reset()

        while not done:
            j = env.turno_actual
            mask = env.get_action_mask(j)
            collector.observe(j, obs, mask)
            accion = seleccionar_accion(policy_net, obs, mask, epsilon, device)
            siguiente_obs, reward, done, _ = env.step(accion)
            collector.record(j, obs, mask, accion, reward * rs)
            global_step += 1

            if done:
                puntos = env.calcular_puntuaciones_finales()
                diff = float(puntos[1] - puntos[0])                     # >0 si gana el jugador 0
                collector.finish({0: tc * diff * rs, 1: -tc * diff * rs}, siguiente_obs)
                hist_pts.append(puntos)

            obs = siguiente_obs

            if len(buffer) >= config['WARMUP_STEPS'] and global_step % config['TRAIN_FREQ'] == 0:
                hist_loss.append(entrenar_step(policy_net, target_net, buffer, optimizer, config['BATCH_SIZE']))
                actualizar_target_suave(policy_net, target_net, config['TAU'])

        if ep % config['LOG_EVERY'] == 0:
            p = np.mean(hist_pts[-config['LOG_EVERY']:], axis=0)
            l = np.mean(hist_loss[-500:]) if hist_loss else float('nan')
            print(f"Ep {ep:>6}/{n_ep} | eps {epsilon:.3f} | pasos {global_step:>8} | "
                  f"pts self-play J0/J1 {p[0]:5.1f}/{p[1]:5.1f} | loss {l:.4f} | {(time.time()-t0)/60:.1f} min")

        if ep % config['EVAL_EVERY'] == 0:
            res = duelo(agente_dqn(policy_net, device), agente_heuristico, config['EVAL_GAMES'])
            imprimir_duelo(f"[ep {ep}] DQN vs heurístico", res)
            torch.save(policy_net.state_dict(), f"{config['CKPT_DIR']}/ep{ep}.pth")
            if res['winrate'] > mejor_wr:
                mejor_wr = res['winrate']
                torch.save(policy_net.state_dict(), f"{config['CKPT_DIR']}/best.pth")
                print(f"   -> nuevo mejor checkpoint (winrate {100*mejor_wr:.1f}%)")

    torch.save(policy_net.state_dict(), config['SAVE_PATH'])
    print(f"Entrenamiento terminado. Modelo final: {config['SAVE_PATH']} | mejor winrate vs heurístico: {100*mejor_wr:.1f}%")
    return policy_net
