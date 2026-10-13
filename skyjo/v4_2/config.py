# ==============================================================================
# train.py  (hiperparámetros + lanzamiento del entrenamiento v4.2)
# ==============================================================================
CONFIG = {
    'NUM_EPISODIOS': 20000,
    'BUFFER_CAPACITY': 200000,
    'BATCH_SIZE': 1024,
    'GAMMA': 0.99,
    'LR': 1e-4,
    'TRAIN_FREQ': 16,
    'EPSILON_START': 1.0,
    'EPSILON_END': 0.05,
    'TARGET_UPDATE_FREQ': 1000,
    'WARMUP_STEPS': 2000,
    'LOG_EVERY': 250,            # frecuencia del log de entrenamiento (como en 5.0)
    'EVAL_EVERY': 2500,          # nuevo en 4.2: duelo vs heurístico cada 2500 episodios
    'EVAL_GAMES': 300,           # partidas por evaluación
    'CKPT_DIR': 'checkpoints_4.2',
    'SAVE_PATH': 'skyjo_dqn_model_4.2.pth'
}
