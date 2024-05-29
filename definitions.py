import os

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))  # This is your Project Root

OUT_DIR = os.path.join(ROOT_DIR, 'out')
CONFIG_DIR = os.path.join(ROOT_DIR, 'config')

LOG_DIR = os.path.join(OUT_DIR, 'log')
os.makedirs(LOG_DIR, exist_ok=True)
