from .env import SkyjoEnv
from .model import SkyjoDQN

import numpy as np
import torch
import random
from tqdm import tqdm

def seleccionar_accion_dqn(model, obs, mask, device="cpu"):
    model.eval()
    with torch.no_grad():
        obs_tensor = {
            'valores_tableros': torch.FloatTensor(obs['valores_tableros']).unsqueeze(0).to(device),
            'visibilidad_tableros': torch.FloatTensor(obs['visibilidad_tableros']).unsqueeze(0).to(device),
            'escalares': torch.FloatTensor(obs['escalares']).unsqueeze(0).to(device)
        }

        q_values = model(obs_tensor)

    mask_tensor = torch.BoolTensor(mask).to(device)
    q_values[0, ~mask_tensor] = -1e9

    return torch.argmax(q_values, dim=1).item()

def seleccionar_accion_aleatoria(mask):
    acciones_validas = np.where(mask)[0]
    return random.choice(acciones_validas)

def evaluar_agente():
    NUM_PARTIDAS = 2000
    MODEL_PATH = "skyjo_dqn_model_4.2.pth"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    env = SkyjoEnv(num_jugadores=2)
    model = SkyjoDQN(num_jugadores=2).to(device)

    try:
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        print(f"¡Modelo '{MODEL_PATH}' cargado con éxito en {device}!\n")
    except FileNotFoundError:
        print(f"Error: No se encontró el archivo '{MODEL_PATH}'. Asegúrate de haber completado el entrenamiento.")
        return

    victorias_dqn = 0
    victorias_random = 0
    empates = 0

    puntos_totales_dqn = 0
    puntos_totales_random = 0
    historial_turnos = []

    print(f"--- Iniciando evaluación de {NUM_PARTIDAS} partidas ---")

    for partida in tqdm(range(1, NUM_PARTIDAS + 1), desc="Evaluando"):
        obs = env.reset()
        done = False
        turnos_partida = 0

        v2_es_jugador_0 = (partida % 2 != 0)

        while not done:
            jugador_actual = env.turno_actual
            mask = env.get_action_mask(jugador_actual)

            if (jugador_actual == 0 and v2_es_jugador_0) or (jugador_actual == 1 and not v2_es_jugador_0):
                accion = seleccionar_accion_dqn(model, obs, mask, device=device)
            else:
                accion = seleccionar_accion_aleatoria(mask)

            obs, reward, done, info = env.step(accion)
            turnos_partida += 1

        historial_turnos.append(turnos_partida)

        puntos = env.calcular_puntuaciones_finales()
        puntos_dqn = puntos[0] if v2_es_jugador_0 else puntos[1]
        puntos_random = puntos[1] if v2_es_jugador_0 else puntos[0]

        puntos_totales_dqn += puntos_dqn
        puntos_totales_random += puntos_random

        if puntos_dqn < puntos_random:
            victorias_dqn += 1
        elif puntos_random < puntos_dqn:
            victorias_random += 1
        else:
            empates += 1

    winrate_dqn = (victorias_dqn / NUM_PARTIDAS) * 100
    winrate_random = (victorias_random / NUM_PARTIDAS) * 100
    pct_empates = (empates / NUM_PARTIDAS) * 100

    media_puntos_dqn = puntos_totales_dqn / NUM_PARTIDAS
    media_puntos_random = puntos_totales_random / NUM_PARTIDAS

    min_turnos = np.min(historial_turnos)/4
    max_turnos = np.max(historial_turnos)/4
    media_turnos = np.mean(historial_turnos)/4

    print("\n" + "="*50)
    print("           RESULTADOS DE LA EVALUACIÓN          ")
    print("="*50)
    print(f" Total Partidas Jugadas: {NUM_PARTIDAS}")
    print(f" Victorias DQN (IA):     {victorias_dqn:4d} ({winrate_dqn:.2f}%)")
    print(f" Victorias Aleatorio:   {victorias_random:4d} ({winrate_random:.2f}%)")
    print(f" Empates:               {empates:4d} ({pct_empates:.2f}%)")
    print("-" * 50)
    print(f" Puntuación Media DQN:       {media_puntos_dqn:.1f} pts (Menos es mejor)")
    print(f" Puntuación Media Aleatorio: {media_puntos_random:.1f} pts")
    print("-" * 50)
    print(f" Turnos Mínimos por partida: {min_turnos}")
    print(f" Turnos Máximos por partida: {max_turnos}")
    print(f" Turnos Medios por partida:  {media_turnos:.1f}")
    print("="*50)

if __name__ == "__main__":
    evaluar_agente()
