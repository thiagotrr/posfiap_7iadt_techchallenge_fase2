import os
import re
from typing import Tuple, Optional
from datetime import datetime
import hashlib

import joblib
import pandas as pd
import kagglehub as kh
from sklearn.preprocessing import MinMaxScaler
import base64
from typing import List


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


def salvar_modelo(modelo, scaler=None, params: Optional[dict] = None, caminho_salvar: Optional[str] = None, id_hex: Optional[str] = None):
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
        # if an id_hex is provided and caminho_salvar is a filename, try to append id before extension
        if id_hex:
            base, ext = os.path.splitext(caminho_salvar)
            caminho_salvar = f"{base}_{id_hex}{ext}"
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

    if id_hex:
        file_name = f'rdm_forest_smote_v{next_version}_{id_hex}.joblib'
    else:
        file_name = f'rdm_forest_smote_v{next_version}.joblib'
    file_path = os.path.join(models_dir, file_name)

    joblib.dump(artefatos, file_path)
    return file_path


def gerar_identificador_hex(dt: Optional[datetime] = None) -> str:
    """Gera um identificador hexadecimal de 6 caracteres usando dia-mes-ano-hora como chave.

    Se `dt` não for fornecido, usa `datetime.now()`.
    """
    if dt is None:
        dt = datetime.now()
    chave = dt.strftime("%Y%m%d_%H%M%S")
    digest = hashlib.md5(chave.encode()).hexdigest()
    return digest[:6]

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


def _localizar_arquivo_por_id(diretorio: str, id_hex: str, extensoes: List[str]) -> Optional[str]:
    """Procura por um arquivo em `diretorio` que contenha o sufixo _{id_hex} antes da extensão.

    Retorna o caminho completo do arquivo encontrado ou None.
    """
    if not os.path.exists(diretorio):
        return None

    for nome in os.listdir(diretorio):
        for ext in extensoes:
            if nome.endswith(f"_{id_hex}{ext}"):
                return os.path.join(diretorio, nome)
    return None


def encode_file_base64(caminho_arquivo: str) -> Optional[dict]:
    """Lê um arquivo e retorna um dicionário com `filename` e `content_base64`.

    Retorna None se o arquivo não existir.
    """
    if not caminho_arquivo or not os.path.exists(caminho_arquivo):
        return None
    with open(caminho_arquivo, "rb") as f:
        dados = f.read()
    conteudo_b64 = base64.b64encode(dados).decode("ascii")
    return {"filename": os.path.basename(caminho_arquivo), "content_base64": conteudo_b64}


def baixar_modelo_base64(id_hex: str) -> Optional[dict]:
    """Localiza um modelo em `src/ml/models` com o sufixo _{id_hex} e retorna conteúdo codificado em base64."""
    models_dir = os.path.join("src", "ml", "models")
    caminho = _localizar_arquivo_por_id(models_dir, id_hex, [".joblib", ".pkl"])
    if not caminho:
        return None
    return encode_file_base64(caminho)


def baixar_log_base64(id_hex: str) -> Optional[dict]:
    """Localiza um log em `src/ml/log` com o sufixo _{id_hex} e retorna conteúdo codificado em base64."""
    logs_dir = os.path.join("src", "ml", "log")
    caminho = _localizar_arquivo_por_id(logs_dir, id_hex, [".log", ".txt"])
    if not caminho:
        return None
    return encode_file_base64(caminho)


def localizar_caminho_modelo(id_hex: str) -> Optional[str]:
    """Retorna o caminho do arquivo de modelo em `src/ml/models` que contenha o sufixo _{id_hex}.

    Usado para downloads diretos via FileResponse.
    """
    models_dir = os.path.join("src", "ml", "models")
    return _localizar_arquivo_por_id(models_dir, id_hex, [".joblib", ".pkl"])


def localizar_caminho_log(id_hex: str) -> Optional[str]:
    """Retorna o caminho do arquivo de log em `src/ml/log` que contenha o sufixo _{id_hex}.

    Usado para downloads diretos via FileResponse.
    """
    logs_dir = os.path.join("src", "ml", "log")
    return _localizar_arquivo_por_id(logs_dir, id_hex, [".log", ".txt"]) 