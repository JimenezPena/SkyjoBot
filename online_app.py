import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import time
import uuid

import torch
torch.set_num_threads(1)

import numpy as np

from flask_socketio import SocketIO, join_room, emit
import string
import random

from flask import Flask, render_template, jsonify, request, session
from skyjo.v4_2 import SkyjoEnv            # motor de juego (las reglas son iguales en todas las versiones)
from skyjo.niveles import cargar_nivel, listar_niveles, niveles_para
from skyjo.vista import construir_vista

app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", "clave-solo-para-local")

device = torch.device("cpu")

# Oponente por defecto (el modelo 4.2). Se puede cambiar con la variable de entorno SKYJO_NIVEL
# o por partida con {"nivel": "..."} en POST /api/reset (opcional: index.html no lo necesita).
NIVEL_POR_DEFECTO = os.environ.get("SKYJO_NIVEL", "dqn_5.1")

_agentes = {}
def get_agente(nivel):
    if nivel not in _agentes:
        _agentes[nivel] = cargar_nivel(nivel, device=device)
    return _agentes[nivel]

# ---- Partidas en memoria: una por visitante ----
games = {}                 # id -> {"env": SkyjoEnv, "nivel": str, "t": último uso}
MAX_EDAD = 60 * 60         # borra partidas inactivas tras 1 hora

def tableros_publicos(env):
    """Tableros sin los valores de las cartas boca abajo."""
    t = env.tableros.copy()
    t[:, :, :, 0] = np.where(t[:, :, :, 1] == 1, t[:, :, :, 0], 0)
    return t.tolist()

def _obtener_partida(almacen, clave, n, nivel):
    ahora = time.time()
    for k in [k for k, g in almacen.items() if ahora - g["t"] > MAX_EDAD]:
        del almacen[k]

    gid = session.get(clave)
    if gid not in almacen:
        gid = uuid.uuid4().hex
        session[clave] = gid
        almacen[gid] = {"env": SkyjoEnv(num_jugadores=n), "nivel": nivel, "t": ahora}
    almacen[gid]["t"] = ahora
    return almacen[gid]

def get_game():
    return _obtener_partida(games, "gid", 2, NIVEL_POR_DEFECTO)

def get_env():
    return get_game()["env"]

def get_state_response(env):
    puntos = env.calcular_puntuaciones_finales() if env.game_over else None
    return {
        "tableros": tableros_publicos(env),
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

@app.route('/api/niveles', methods=['GET'])
def niveles():
    return jsonify({"niveles": list(listar_niveles()), "por_defecto": NIVEL_POR_DEFECTO})

@app.route('/api/reset', methods=['POST'])
def reset_game():
    game = get_game()
    nivel = (request.get_json(silent=True) or {}).get('nivel')
    if nivel is not None:
        if nivel not in listar_niveles():
            return jsonify({"error": "Nivel desconocido"}), 400
        game["nivel"] = nivel
    env = game["env"]
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

    mask = env.get_action_mask(0)
    if not isinstance(accion, int) or not (0 <= accion < len(mask)) or not mask[accion]:
        return jsonify({"error": "Acción no permitida"}), 400

    env.step(accion)
    return jsonify(get_state_response(env))

@app.route('/api/ia_step', methods=['POST'])
def ia_step():
    game = get_game()
    env = game["env"]
    if env.game_over:
        return jsonify({"error": "Partida terminada"}), 400
    if env.turno_actual != 1:
        return jsonify({"error": "No es el turno de la IA"}), 400

    vista = construir_vista(env, 1)                  # el agente solo ve la vista pública
    mask = vista.mask
    accion_ia = int(get_agente(game["nivel"])(vista))
    if not (0 <= accion_ia < len(mask)) or not mask[accion_ia]:      # red de seguridad: nunca bloquear la partida
        accion_ia = int(np.random.choice(np.flatnonzero(mask)))
    env.step(accion_ia)
    return jsonify(get_state_response(env))

# ---- Modo 4 jugadores: tú (asiento 0) contra tres bots (asientos 1, 2 y 3, en el orden de turno) ----
# Los bots solo ven la vista pública. Por defecto son heurísticos (SKYJO_NIVEL_4); los DQN actuales son de 2 jugadores.
NIVEL_4J = os.environ.get("SKYJO_NIVEL_4", "dqn_5.1")#"heuristico")
games4 = {}

def get_game4():
    game = _obtener_partida(games4, "gid4", 4, NIVEL_4J)
    nivel = game["nivel"] if game["nivel"] in niveles_para(4) else "heuristico"   # p. ej. un DQN de 2 jugadores no vale aquí
    game.setdefault("niveles", [nivel] * 3)          # nivel de cada bot (asientos 1, 2 y 3)
    return game

def respuesta4(game):
    r = get_state_response(game["env"])
    r["niveles"] = game["niveles"]
    return r

@app.route('/api/4/niveles', methods=['GET'])
def niveles4():
    validos = niveles_para(4)
    return jsonify({"niveles": validos, "por_defecto": NIVEL_4J if NIVEL_4J in validos else "heuristico"})

@app.route('/cuatro')
def cuatro():
    return render_template('cuatro.html')

@app.route('/api/4/reset', methods=['POST'])
def reset_game4():
    game = get_game4()
    datos = request.get_json(silent=True) or {}
    niveles = datos.get('niveles') or ([datos['nivel']] * 3 if datos.get('nivel') else None)
    if niveles is not None:
        validos = niveles_para(4)
        if not isinstance(niveles, list) or len(niveles) != 3 or any(x not in validos for x in niveles):
            return jsonify({"error": "Niveles no válidos para 4 jugadores", "validos": validos}), 400
        game["niveles"] = list(niveles)
    game["env"].reset()
    return jsonify(respuesta4(game))

@app.route('/api/4/step', methods=['POST'])
def step_game4():
    game = get_game4()
    env = game["env"]
    accion = (request.get_json(silent=True) or {}).get('action')
    if env.game_over:
        return jsonify({"error": "Partida terminada"}), 400
    if env.turno_actual != 0:
        return jsonify({"error": "No es tu turno"}), 400
    mask = env.get_action_mask(0)
    if not isinstance(accion, int) or not (0 <= accion < len(mask)) or not mask[accion]:
        return jsonify({"error": "Acción no permitida"}), 400
    env.step(accion)
    return jsonify(respuesta4(game))

@app.route('/api/4/bot_step', methods=['POST'])
def bot_step4():
    game = get_game4()
    env = game["env"]
    if env.game_over:
        return jsonify({"error": "Partida terminada"}), 400
    if env.turno_actual == 0:
        return jsonify({"error": "Es tu turno"}), 400
    vista = construir_vista(env, env.turno_actual)
    mask = vista.mask
    accion = int(get_agente(game["niveles"][env.turno_actual - 1])(vista))
    if not (0 <= accion < len(mask)) or not mask[accion]:
        accion = int(np.random.choice(np.flatnonzero(mask)))
    env.step(accion)
    return jsonify(respuesta4(game))

# ---- Modo 2 jugadores humanos: tú (asiento 0) y rival (asiento 1)

# Inicializar SocketIO integrado con tu app de Flask
socketio = SocketIO(app, cors_allowed_origins="*")

# Almacén para las partidas multijugador entre humanos (Salas de 2 jugadores)
# Estructura: { "ABCD": { "env": SkyjoEnv(2), "jugadores": {0: sid_p1, 1: sid_p2} } }
multi_rooms = {}

def generar_codigo_sala():
    return ''.join(random.choices(string.ascii_uppercase + string.digits, k=4))

@app.route('/multijugador')
def pagina_multijugador():
    return render_template('multijugador.html')

@socketio.on('crear_partida')
def handle_crear_partida(data=None):
    room = generar_codigo_sala()
    while room in multi_rooms:
        room = generar_codigo_sala()
    
    env = SkyjoEnv(num_jugadores=2)
    env.reset()
    
    multi_rooms[room] = {
        "env": env,
        "jugadores": {0: request.sid, 1: None}
    }
    join_room(room)
    emit('partida_creada', {"room": room, "asiento": 0})
    emit_estado_multijugador(room)

@socketio.on('unirse_partida')
def handle_unirse_partida(data=None):
    if not data:
        emit('error_multijugador', {"mensaje": "Datos de sala no recibidos"})
        return
        
    room = data.get('room', '').upper()
    if room in multi_rooms and multi_rooms[room]["jugadores"][1] is None:
        multi_rooms[room]["jugadores"][1] = request.sid
        join_room(room)
        emit('unido_exito', {"room": room, "asiento": 1})
        emit_estado_multijugador(room)
    else:
        emit('error_multijugador', {"mensaje": "Sala no encontrada o ya está llena"})

@socketio.on('accion_jugador')
def handle_accion_jugador(data=None):
    if not data:
        return
        
    room = data.get('room')
    asiento = data.get('asiento')
    accion = data.get('action')

    if room not in multi_rooms:
        return
    
    game = multi_rooms[room]
    env = game["env"]

    if env.game_over or env.turno_actual != asiento:
        return

    mask = env.get_action_mask(asiento)
    if not isinstance(accion, int) or not (0 <= accion < len(mask)) or not mask[accion]:
        return

    env.step(accion)
    emit_estado_multijugador(room)

def emit_estado_multijugador(room):
    game = multi_rooms[room]
    env = game["env"]
    puntos = env.calcular_puntuaciones_finales() if env.game_over else None
    
    estado_base = {
        "tableros": tableros_publicos(env),
        "turno_actual": int(env.turno_actual),
        "fase_turno": int(env.fase_turno),
        "origen_robo": env.origen_robo,
        "carta_en_mano": int(env.carta_en_mano) if env.carta_en_mano is not None else None,
        "top_descarte": int(env.descartes[-1]) if len(env.descartes) > 0 else None,
        "game_over": env.game_over,
        "puntos": puntos
    }

    # Enviar a cada jugador su tablero y máscara específica de forma privada
    for asiento, sid in game["jugadores"].items():
        if sid:
            estado_jugador = estado_base.copy()
            estado_jugador["action_mask"] = env.get_action_mask(asiento).tolist()
            estado_jugador["mi_asiento"] = asiento
            socketio.emit('actualizar_estado', estado_jugador, room=sid)
            
if __name__ == '__main__':
    #app.run(debug=True, port=5000)
    socketio.run(app, host='0.0.0.0', port=5000, debug=True)
