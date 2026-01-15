import os
import re
from typing import Tuple, Optional

import joblib
import pandas as pd
import kagglehub as kh
from sklearn.preprocessing import MinMaxScaler


def carregar_dataset() -> Tuple[pd.DataFrame, pd.Series, MinMaxScaler]:
    """Carrega e aplica pipeline de tratamento do dataset de doença hepática.

    Retorna X (features normalizadas), y (target) e o `MinMaxScaler` utilizado.
    """
    path = kh.dataset_download("jeevannagaraj/indian-liver-patient-dataset")
    df = pd.read_csv(os.path.join(path, "Indian Liver Patient Dataset (ILPD).csv"))
    df = df.dropna(subset=["alkphos"])
    df["is_patient"] = df["is_patient"].replace(2, 0)
    df["is_patient"] = df["is_patient"].astype(bool)

    principais = ["age", "tot_bilirubin", "direct_bilirubin", "tot_proteins", "albumin", "ag_ratio", "sgpt", "sgot", "alkphos"]

    scaler = MinMaxScaler()
    X = df[principais].copy()
    X[principais] = scaler.fit_transform(X[principais])
    y = df["is_patient"].copy()

    return X, y, scaler


def salvar_modelo(modelo, scaler=None, params: Optional[dict] = None, caminho_salvar: Optional[str] = None):
    """Salva o modelo.

    Se `caminho_salvar` for passado, salva diretamente nesse caminho incluindo `params` e `scaler` quando disponíveis.
    Caso contrário, aplica a lógica de versionamento existente (pasta `src/ml/models`) e salva artefatos com `model` e `scaler`.
    """
    models_dir = os.path.join('src', 'ml', 'models')
    os.makedirs(models_dir, exist_ok=True)

    if caminho_salvar:
        artefatos = {"model": modelo}
        if scaler is not None:
            artefatos["scaler"] = scaler
        if params is not None:
            artefatos["params"] = params
        joblib.dump(artefatos, caminho_salvar)
        return caminho_salvar

    # Lógica de versionamento quando caminho não é informado
    files = os.listdir(models_dir)
    versions = []
    for f in files:
        match = re.match(r'rdm_forest_smote_v(\d+)\.joblib', f)
        if match:
            versions.append(int(match.group(1)))

    if not versions:
        next_version = 0
    else:
        next_version = max(versions) + 1

    artefatos = {"model": modelo}
    if scaler is not None:
        artefatos["scaler"] = scaler
    if params is not None:
        artefatos["params"] = params

    file_name = f'rdm_forest_smote_v{next_version}.joblib'
    file_path = os.path.join(models_dir, file_name)

    joblib.dump(artefatos, file_path)
    return file_path

def validar_existencia_modelo():
    """
    Verifica se existe algum modelo treinado na pasta 'src/ml/models'.

    Retorna:
        bool: True se existir um modelo, False caso contrário.
    """
    models_dir = os.path.join('src', 'ml', 'models')
    if not os.path.exists(models_dir):
        return False

    files = os.listdir(models_dir)
    for f in files:
        if f.endswith('.joblib'):
            return True
    
    return False