import os

MODELS_DIR = "models"
DATA_HIST_DIR = os.path.join("data", "history")
PLOTS_HIST_DIR = os.path.join("plots", "history")

def makedir(dirpath: str) -> str:
    if len(os.path.split(dirpath)) > 1:
        directory = os.path.dirname(dirpath)
    else:
        directory = dirpath
    os.makedirs(directory, exist_ok=True)
    return dirpath
