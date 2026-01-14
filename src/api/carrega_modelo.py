import os
import joblib
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent # src/
MODELS_DIR = BASE_DIR / "ml" / "models"


_artefatos = None

def carrega_modelo():
    global _artefatos
    if _artefatos is not None:
        return _artefatos   
    
    if not MODELS_DIR.exists():
        raise FileNotFoundError(f"Diretório de modelos não encontrado: {MODELS_DIR}")

    files = sorted(
        [ f for f in os.listdir(MODELS_DIR) if f.endswith(".joblib")],
    )

    if not files:
        raise FileNotFoundError("Nenhum modelo treinado encontrado!")
    
    latest = files[-1]
    path = MODELS_DIR / latest

    _artefatos = joblib.load(path)
    return _artefatos