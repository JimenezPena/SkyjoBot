# ============================================================
# Configuración y lanzamiento (v5.0)
# ============================================================
CONFIG = {
    'NUM_EPISODIOS': 20000,
    'BUFFER_CAPACITY': 200_000,
    'BATCH_SIZE': 256,          # antes 1024
    'TRAIN_FREQ': 8,            # ratio de replay = 256/8 = 32 muestras/paso (antes 64)
    'LR': 1e-4,
    'GAMMA': 0.997,
    'N_STEP': 4,                # fase 1 y fase 2 cuentan como pasos -> ~2 turnos
    'TAU': 0.005,               # soft update del target (sustituye al hard sync cada 1000 pasos)
    'REWARD_SCALE': 1 / 25,     # normaliza shaping y recompensa terminal
    'TERMINAL_COEF': 1.5,
    'EPSILON_START': 1.0,
    'EPSILON_END': 0.05,
    'EPS_DECAY_FRAC': 0.80,
    'WARMUP_STEPS': 5000,
    'LOG_EVERY': 250,
    'EVAL_EVERY': 2500,
    'EVAL_GAMES': 300,
    'CKPT_DIR': 'checkpoints_5.0',
    'SAVE_PATH': 'skyjo_dqn_model_5.0.pth',
}
