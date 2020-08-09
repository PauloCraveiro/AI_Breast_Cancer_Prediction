import os
from config import *


def structureCheck():
    # Define corretamente os caminhos do dataset
    APP_ROOT = os.path.dirname(os.path.abspath(__file__))

    DATASET_PATH = os.path.join(APP_ROOT, DATASET_FOLDER)

    # Define corretamente os caminhos dos modelos
    MODELS_PATH = os.path.join(APP_ROOT, MODELS_FOLDER)
    MODEL_EXISTS = os.path.exists(MODELS_PATH)

    DATASET_FILE = os.path.join(APP_ROOT, DATASET_FOLDER, DATA_NAME)

    return APP_ROOT, DATASET_PATH, MODELS_PATH, MODEL_EXISTS, DATASET_FILE