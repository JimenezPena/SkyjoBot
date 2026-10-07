from TrainingFolder.trainer import entrenar_skyjo

CONFIG = {
    'NUM_EPISODIOS': 20000,
    'BUFFER_CAPACITY': 200000,
    'BATCH_SIZE': 512,
    'GAMMA': 0.99,
    'LR': 1e-4,
    'TRAIN_FREQ': 8,
    'EPSILON_START': 1.0,
    'EPSILON_END': 0.05,
    'TARGET_UPDATE_FREQ': 1000,
    'WARMUP_STEPS': 2000,
    'SAVE_PATH': 'skyjo_dqn_model_3.0.pth'
}

if __name__ == "__main__":
    entrenar_skyjo(CONFIG)
