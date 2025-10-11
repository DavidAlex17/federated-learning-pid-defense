import os
import yaml

# Directory this file lives in (…/cfg)
BASE_DIR = os.path.dirname(__file__)

def load(path=None):
    """
    Load project configuration and resolve relative paths.
    If path is None, loads cfg/project.yaml next to this file.
    """
    if path is None:
        path = os.path.join(BASE_DIR, "project.yaml")

    with open(path, "r") as f:
        cfg = yaml.safe_load(f) or {}

    for key in ("data_dir", "results_dir", "plots_dir"):
        if key in cfg:
            cfg[key] = os.path.abspath(cfg[key])

    return cfg
