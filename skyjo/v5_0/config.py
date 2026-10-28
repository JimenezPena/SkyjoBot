# ==============================================================================
# Hiperparámetros de v5.0. Receta base = 4.2 (la que aprendió: BATCH 1024, TRAIN_FREQ 16, target duro cada
# 1000 pasos, GAMMA 0.99, LR 1e-4), pero con partidas de JUGADORES asientos y una red compartida por todos.
# Con JUGADORES=2 el entrenamiento equivale al de 4.2 (misma recompensa), y sirve de comprobación.
# ==============================================================================
CONFIG = {
    'JUGADORES': 4,
    'NUM_EPISODIOS': 20000,
    'BUFFER_CAPACITY': 400000,
    'BATCH_SIZE': 1024,
    'GAMMA': 0.99,
    'LR': 1e-4,
    'TRAIN_FREQ': 16,
    'EPSILON_START': 1.0,
    'EPSILON_END': 0.05,
    'TARGET_UPDATE_FREQ': 1000,
    'WARMUP_STEPS': 2000,
    'RECOMPENSA_FINAL': 1.5,     # recompensa final = 1.5 * (media de puntos de los rivales - mis puntos)
    'RECOMPENSA_VICTORIA': 20.0,  # extra por ganar: V * (victoria - 1/n); prueba 15-30 (los puntos suelen valer ~±20)
    'RECOMPENSA_REF': 'media',   # referencia de la recompensa final: 'media' = mejor que la media de rivales; 'mejor' = solo positiva si ganas (con 'mejor' baja RECOMPENSA_VICTORIA)
    'BARAJAS_SIMETRICAS': True,  # con RIVALES: n episodios seguidos con la misma baraja y el aprendiz rotando por los n asientos
    'SAVE_EVERY': 1000,
    'LOG_EVERY': 500,
    'EVAL_EVERY': 1000,          # duelo vs heurísticos cada 2500 episodios
    'EVAL_GAMES': 300,           # partidas por evaluación = barajas x nº de asientos (cada baraja se juega con el DQN en todos los asientos); el mejor checkpoint = menos puntos medios
    'EVAL_SEMILLA': 12345,       # mismos repartos en todas las evaluaciones (comparación pareada entre checkpoints); None = azar
    'EVAL_JUGADORES': [4, 2],    # nº de asientos a evaluar; el primero decide el "mejor checkpoint"
    'CKPT_DIR': 'checkpoints_5.1',
    'SAVE_PATH': 'skyjo_dqn_model_5.1.pth',
    'IMIT_EPISODIOS': 1200, # parametros para imitar partidas del heurístico para comenzar
    'IMIT_EPOCHS': 12,
    'IMIT_BATCH': 1024,
    'IMIT_LR': 1e-3,
    'EPSILON_START_IMIT': 0.15,
    'CALIBRATION_PRESTEPS': 1000,
    # --- DQfD opcional: pérdida de margen sobre demos durante la fase DQN ---
    'DEMO_LAMBDA': 0.05,          # 0 = desactivado. Prueba 0.05; rango útil 0.01-0.1
    'DEMO_STEPS': 100000,        # primeros pasos de DQN con la pérdida de margen
    'DEMO_MARGIN': 0.8,          # margen m (mismo valor que usaba el paper de DQfD)
    'DEMO_BATCH': 512,           # batch de demos por actualización
    # --- Rivales (ver rivales.py): None = self-play puro (todos los asientos aprenden) ---
    'RIVALES': {'heuristico': 0.7, 'actual': 0.15, 'checkpoint': 0.15},   # pesos para sortear cada asiento rival
    'RIVALES_FIN': None,         # p. ej. {'heuristico': 0.3, 'actual': 0.4, 'checkpoint': 0.3}: interpolación lineal
    'RIVALES_CKPT': [],          # modelos externos como rivales fijos, p. ej. ['modelos/v5_1.pth']
    'ASIENTOS_APRENDIZ': 1,      # asientos (al azar) que aprenden en cada partida; JUGADORES = como self-puro
    'POOL_EVERY': 1000,          # episodios entre instantáneas de la red para el pool de checkpoints
    'POOL_MAX': 5,               # instantáneas que se conservan
    'RIVAL_EPSILON': 0.05,       # ruido de los rivales basados en red
}
