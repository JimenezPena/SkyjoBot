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
    'LOG_EVERY': 250,
    'EVAL_EVERY': 250,          # duelo vs heurísticos cada 2500 episodios
    'EVAL_GAMES': 300,           # partidas por evaluación (>= 1000 para conclusiones fiables)
    'EVAL_JUGADORES': [4, 2],    # nº de asientos a evaluar; el primero decide el "mejor checkpoint"
    'CKPT_DIR': 'checkpoints_5.0',
    'SAVE_PATH': 'skyjo_dqn_model_5.1.pth',
    'IMIT_EPISODIOS': 2500, # parametros para imitar partidas del heurístico para comenzar
    'IMIT_EPOCHS': 20,
    'IMIT_BATCH': 1024,
    'IMIT_LR': 1e-3,
    'EPSILON_START_IMIT': 0.3,
    # --- DQfD opcional: pérdida de margen sobre demos durante la fase DQN ---
    'DEMO_LAMBDA': 0.05,          # 0 = desactivado. Prueba 0.05; rango útil 0.01-0.1
    'DEMO_STEPS': 100000,        # primeros pasos de DQN con la pérdida de margen
    'DEMO_MARGIN': 0.8,          # margen m (mismo valor que usaba el paper de DQfD)
    'DEMO_BATCH': 512,           # batch de demos por actualización
}
