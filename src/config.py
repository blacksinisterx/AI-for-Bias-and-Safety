from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT_DIR / "train.csv"
SPLIT_PATH = ROOT_DIR / "artifacts" / "splits" / "split_indices.json"
MODELS_DIR = ROOT_DIR / "artifacts" / "models"
CACHE_DIR = ROOT_DIR / "artifacts" / "cache"

RANDOM_STATE = 42
TRAIN_SIZE = 100_000
EVAL_SIZE = 20_000
MAX_LENGTH = 128
BASE_MODEL_NAME = "distilbert-base-uncased"
