import kagglehub as kh
import pandas as pd
import os
import re
import joblib
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import make_scorer, accuracy_score, f1_score, recall_score
from imblearn.over_sampling import SMOTE

def treinar_modelo():
    """
    Função para treinar o modelo de classificação de pacientes com doença hepática.

    A função realiza as seguintes etapas:
    1. Baixa o dataset do Kaggle.
    2. Realiza o pré-processamento dos dados, incluindo limpeza, normalização e encoding.
    3. Divide os dados em conjuntos de treino e teste.
    4. Realiza uma busca em grade (GridSearch) para encontrar os melhores hiperparâmetros para o RandomForestClassifier.
    5. Aplica a técnica SMOTE para lidar com o desbalanceamento de classes.
    6. Treina o modelo final com os melhores parâmetros e os dados rebalanceados.
    7. Avalia o modelo usando acurácia, F1-score e validação cruzada.

    Retorna:
        tuple: Uma tupla contendo:
            - RandomForestClassifier: O objeto do modelo treinado.
            - pd.DataFrame: Um DataFrame com as métricas de validação (acurácia, F1-score, score da validação cruzada).
    """
    path = kh.dataset_download("jeevannagaraj/indian-liver-patient-dataset")
    print(f"Caminho para os arquivos: {path}")
    df = pd.read_csv(os.path.join(path, "Indian Liver Patient Dataset (ILPD).csv"))

    df = df.dropna(subset=['alkphos'])

    df['is_patient'] = df['is_patient'].replace(2, 0)
    df['is_patient'] = df['is_patient'].astype(bool)

    dataset = df.copy()

    main_features = ['age', 'tot_bilirubin', 'direct_bilirubin', 'tot_proteins', 'albumin', 'ag_ratio', 'sgpt', 'sgot', 'alkphos']

    scaler = MinMaxScaler() #chamando o metodo de normalização dos dados (0-1)
    dataset_minmax_scaler = dataset.copy()

    dataset_minmax_scaler[main_features] = scaler.fit_transform(dataset_minmax_scaler[main_features])

    X = dataset_minmax_scaler[main_features]
    y = dataset_minmax_scaler['is_patient']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, stratify=y, random_state=42)


    # Primeiros parâmetros testados
    # param_grid = { "n_estimators":[12,25,50,100],
    #     "max_depth":[2,3,5,7],
    #     "random_state":[3,5,7,9,None],
    #     "criterion": ["gini", "entropy"],
    #     "class_weight": ["balanced", "balanced_subsample"]
    # }

    param_grid = { "n_estimators":[25,50,100,200], # Maior floresta
        "max_depth":[5,7,9, None], # Maior profundidade
        "min_samples_split":[2,5,10], # Considerar divisões
        "min_samples_leaf":[1,2,4] # Considerar números de folhas
    }

    #gs: GridSearch
    gs_metric_accuracy = make_scorer(accuracy_score, greater_is_better=True)
    gs_metric_f1_score = make_scorer(f1_score, greater_is_better=True)
    gs_metric_recall = make_scorer(recall_score, greater_is_better=True)

    grid = GridSearchCV(
        RandomForestClassifier(),
        param_grid,
        scoring={'accuracy': gs_metric_accuracy, 'f1': gs_metric_f1_score, 'recall': gs_metric_recall},
        refit='f1',
        cv=5,
        n_jobs=4,
        verbose = 3
    )
    grid.fit(X_train, y_train)

    
    # Na Fase 1, imprimimos os melhores parâmetros encontrados e colocamos explicitamente
    # rf = RandomForestClassifier(class_weight='balanced', criterion='gini', max_depth=5, n_estimators=100, min_samples_leaf = 2, min_samples_split = 10)
    rf = grid.best_estimator_

    # Aplicando SMOTE
    sm = SMOTE()
    X_train_resample, y_train_resample = sm.fit_resample(X_train, y_train)

    rf.fit(X_train_resample, y_train_resample)
    y_predito_random_forest_smote = rf.predict(X_test)

    # Validação do Modelo: recuperando acurácia, f1-score e recall
    accuracy = accuracy_score(y_test, y_predito_random_forest_smote)
    f1score_value = f1_score(y_test, y_predito_random_forest_smote,  average='binary')
    recall_value = recall_score(y_test, y_predito_random_forest_smote, average='binary')

    # Cross Validation
    kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_score = cross_val_score(rf, X_train_resample, y_train_resample, cv=kfold)
    
    validations = {
        "accuracy": accuracy,
        "f1_score": f1score_value,
        "recall": recall_value,
        "cv_score": cv_score.mean()
    }

    df_random_forest_smote = pd.DataFrame(validations, index=["RandomForest SMOTE"])
    
    return rf, scaler, df_random_forest_smote

def salvar_modelo(modelo, scaler):
    """
    Salva o modelo Scikit-Learn treinado em um arquivo versionado usando joblib.

    Args:
        modelo (sklearn.base.BaseEstimator): O objeto do modelo treinado a ser salvo.
    """
    models_dir = os.path.join('src', 'ml', 'models')
    os.makedirs(models_dir, exist_ok=True)

    # Lógica de versionamento
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

    artefatos = {
        "model": modelo,
        "scaler": scaler
    }

    file_name = f'rdm_forest_smote_v{next_version}.joblib'
    file_path = os.path.join(models_dir, file_name)

    joblib.dump(artefatos, file_path)
    print(f"Modelo salvo com sucesso em: {file_path}")

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

if __name__ == "__main__":
    print("Iniciando o treinamento do modelo...")
    modelo_treinado, scaler_treinado, df_metricas = treinar_modelo()
    
    print("\nMetricas de validacao do modelo:")
    print(df_metricas)
    print("\nModelo treinado com sucesso!")
    
    print("\nSalvando o modelo treinado...")
    salvar_modelo(modelo_treinado, scaler_treinado)