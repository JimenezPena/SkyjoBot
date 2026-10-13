# ============================================================
# Utilidades de acción (v5.0)
# ============================================================
import random
import numpy as np
import torch


def obs_a_tensor(obs, device="cpu"):
    return {k: torch.from_numpy(np.asarray(v, dtype=np.float32)).unsqueeze(0).to(device)
            for k, v in obs.items()}


@torch.no_grad()
def accion_greedy(model, obs, mask, device="cpu"):
    m = torch.from_numpy(mask).unsqueeze(0).to(device)
    q = model(obs_a_tensor(obs, device), m).masked_fill(~m, -1e9)
    return int(q.argmax(dim=1).item())


def seleccionar_accion(model, obs, mask, epsilon, device="cpu"):
    if random.random() < epsilon:
        return int(random.choice(np.flatnonzero(mask)))
    return accion_greedy(model, obs, mask, device)
