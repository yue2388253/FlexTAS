import os

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))  # This is your Project Root

OUT_DIR = os.path.join(ROOT_DIR, 'out')
os.makedirs(OUT_DIR, exist_ok=True)
