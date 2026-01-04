import yaml
import os
from pathlib import Path

project_root = Path(__file__).parent.parent
config_path = project_root / 'config.yaml'

with open(config_path, 'r', encoding='utf-8') as f:
    config = yaml.safe_load(f)

data_dir = project_root / config['data']['data_dir']
processed_dir = project_root / config['data']['processed_dir']

data_dir.mkdir(parents=True, exist_ok=True)
processed_dir.mkdir(parents=True, exist_ok=True)
