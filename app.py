from flask import Flask, render_template, jsonify, request
import torch
import numpy as np

# Cargar el entorno y el modelo desde sus respectivos archivos
from skyjo_env import SkyjoEnv
from skyjo_model import SkyjoDQN, seleccionar_accion_dqn

app = Flask(__name__)

# Cargar entorno y modelo globalmente
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
env = SkyjoEnv(num_jugadores=2)

model = SkyjoDQN(num_jugadores=2).to(device)
try:
    model.load_state_dict(torch.load("skyjo_dqn_model_2.0.pth", map_location=device))
    model.eval()
    print("🤖 Modelo IA cargado en el servidor Web.")
except Exception as e:
    print(f"⚠️ No se pudo cargar el modelo IA: {e}")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/reset', methods=['POST'])
def reset_game():
    obs = env.reset()
    return jsonify(get_state_response())

@app.route('/api/step', methods=['POST'])
def step_game():
    data = request.json
    accion = data.get('action')

    if env.game_over:
        return jsonify({"error": "Partida terminada"}), 400
    if env.turno_actual != 0:
        return jsonify({"error": "No es tu turno"}), 400

    # Solo se ejecuta la acción del humano; la IA se gestiona en /api/ia_step
    env.step(accion)
    return jsonify(get_state_response())


@app.route('/api/ia_step', methods=['POST'])
def ia_step():
    """Ejecuta UNA sola acción de la IA (robar o colocar/descartar)."""
    if env.game_over:
        return jsonify({"error": "Partida terminada"}), 400
    if env.turno_actual != 1:
        return jsonify({"error": "No es el turno de la IA"}), 400

    obs = env.obtener_observacion(1)
    mask = env.get_action_mask(1)
    accion_ia = seleccionar_accion_dqn(model, obs, mask, device=device)
    env.step(accion_ia)

    return jsonify(get_state_response())
    
def get_state_response():
    """Genera la estructura de datos que necesita la interfaz web."""
    puntos = None
    if env.game_over:
        puntos = env.calcular_puntuaciones_finales()

    return {
        "tableros": env.tableros.tolist(), # [num_jugadores, 3, 4, 2]
        "turno_actual": int(env.turno_actual),
        "fase_turno": int(env.fase_turno),
        "origen_robo": env.origen_robo,
        "carta_en_mano": int(env.carta_en_mano) if env.carta_en_mano is not None else None,
        "top_descarte": int(env.descartes[-1]) if len(env.descartes) > 0 else None,
        "game_over": env.game_over,
        "puntos": puntos,
        "action_mask": env.get_action_mask(env.turno_actual).tolist()
    }

if __name__ == '__main__':
    app.run(debug=True, port=5000)
