import os

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASE_DIR = os.path.dirname(PROJECT_DIR)
DATASETS_DIR = os.path.join(BASE_DIR, 'docker', 'volumes', 'grafana', 'datasets')
CHECKPOINTS_DIR = os.path.join(BASE_DIR, 'checkpoints')

if not os.path.exists(DATASETS_DIR):
    os.makedirs(DATASETS_DIR, exist_ok=True)
