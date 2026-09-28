import random
import numpy as np
import torch

def seleccionar_accion(model, observacion, env, epsilon, device="cpu"):
    mask = env.get_action_mask(env.turno_actual)
    acciones_validas = np.where(mask)[0]

    if random.random() < epsilon:
        return random.choice(acciones_validas)

    model.eval()
    with torch.no_grad():
        obs_tensor = {
            'valores_tableros': torch.FloatTensor(observacion['valores_tableros']).unsqueeze(0).to(device),
            'visibilidad_tableros': torch.FloatTensor(observacion['visibilidad_tableros']).unsqueeze(0).to(device),
            'escalares': torch.FloatTensor(observacion['escalares']).unsqueeze(0).to(device)
        }
        q_values = model(obs_tensor)

    mask_tensor = torch.BoolTensor(mask).to(device)
    q_values[0, ~mask_tensor] = -1e9

    return torch.argmax(q_values, dim=1).item()
