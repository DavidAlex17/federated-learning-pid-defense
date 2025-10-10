import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from cfg.load_config import load as load_cfg

CFG = load_cfg()
print("Loaded configuration:", CFG)