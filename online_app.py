import os
import time
import uuid

import torch
from flask import Flask, render_template, jsonify, request, session

from skyjo_env import SkyjoEnv
from skyjo_model import SkyjoDQN, seleccionar_accion_dqn

app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", "clave-solo-para-local")

device = torch.device("cpu")   # los servidores gratuitos no tienen GPU
model = SkyjoDQN(num_jugadores=2).to(device)
model.load_state_dict(torch.load("skyjo_dqn_model_2.0.pth", map_location=device))
model.eval()

# ---- Partidas en memoria: una por visitante ----
games = {}                 # id -> {"env": SkyjoEnv, "t": último uso}
MAX_EDAD = 60 * 60         # borra partidas inactivas tras 1 hora

def get_env():
    ahora = time.time()
    for k in [k for k, g in games.items() if ahora - g["t"] > MAX_EDAD]:
        del games[k]

    gid = session.get("gid")
    if gid not in games:
        gid = uuid.uuid4().hex
        session["gid"] = gid
        games[gid] = {"env": SkyjoEnv(num_jugadores=2), "t": ahora}
    games[gid]["t"] = ahora
    return games[gid]["env"]

def get_state_response(env):
    puntos = env.calcular_puntuaciones_finales() if env.game_over else None
    return {
        "tableros": env.tableros.tolist(),
        "turno_actual": int(env.turno_actual),
        "fase_turno": int(env.fase_turno),
        "origen_robo": env.origen_robo,
        "carta_en_mano": int(env.carta_en_mano) if env.carta_en_mano is not None else None,
        "top_descarte": int(env.descartes[-1]) if len(env.descartes) > 0 else None,
        "game_over": env.game_over,
        "puntos": puntos,
        "action_mask": env.get_action_mask(env.turno_actual).tolist()
    }

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/reset', methods=['POST'])
def reset_game():
    env = get_env()
    env.reset()
    return jsonify(get_state_response(env))

@app.route('/api/step', methods=['POST'])
def step_game():
    env = get_env()
    accion = request.json.get('action')

    if env.game_over:
        return jsonify({"error": "Partida terminada"}), 400
    if env.turno_actual != 0:
        return jsonify({"error": "No es tu turno"}), 400

    env.step(accion)
    return jsonify(get_state_response(env))

@app.route('/api/ia_step', methods=['POST'])
def ia_step():
    env = get_env()
    if env.game_over:
        return jsonify({"error": "Partida terminada"}), 400
    if env.turno_actual != 1:
        return jsonify({"error": "No es el turno de la IA"}), 400

    obs = env.obtener_observacion(1)
    mask = env.get_action_mask(1)
    accion_ia = seleccionar_accion_dqn(model, obs, mask, device=device)
    env.step(accion_ia)
    return jsonify(get_state_response(env))

if __name__ == '__main__':
    app.run(debug=True, port=5000)
