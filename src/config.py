# src/config.py

import os

# Project paths
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(PROJECT_DIR, '..', 'data')
MODELS_DIR = os.path.join(PROJECT_DIR, '..', 'models')
RESULTS_DIR = os.path.join(PROJECT_DIR, '..', 'results')
CONFIG_DIR = os.path.join(PROJECT_DIR, '..', 'config')

# Data paths
TAXI_DATA_PATH = os.path.join(DATA_DIR, '2021_Yellow_Taxi_Trip_Data_20240406.csv')

# Hyperparameters
HYPERPARAMS_FILE = os.path.join(CONFIG_DIR, 'hyperparams.yml')

# Create directories if they don't exist
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(CONFIG_DIR, exist_ok=True)