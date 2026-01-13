import kagglehub as kh
import pandas as pd
import os
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import export_graphviz
from sklearn.metrics import make_scorer, accuracy_score, f1_score
from imblearn.over_sampling import SMOTE



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

grid = GridSearchCV(RandomForestClassifier(), param_grid, scoring={'accuracy': gs_metric_accuracy, 'f1': gs_metric_f1_score}, refit='f1', cv=5, n_jobs=4, verbose = 3)
grid.fit(X_train, y_train)

random_forest_params = grid.best_params_

rf = RandomForestClassifier(class_weight='balanced', criterion='gini', max_depth=5, n_estimators=100, min_samples_leaf = 2, min_samples_split = 10)

# Aplicando SMOTE
sm = SMOTE()
X_train_resample, y_train_resample = sm.fit_resample(X, y)

rf.fit(X_train_resample, y_train_resample)
y_predito_random_forest_smote = rf.predict(X_test)

# Validação do Modelo: recuperando acurácia e f1-score
accuracy = accuracy_score(y_test, y_predito_random_forest_smote)
f1score_value = f1_score(y_test, y_predito_random_forest_smote,  average='binary')

# Cross Validation
kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_score = cross_val_score(rf, X_train_resample, y_train_resample, cv=kfold)
print(cv_score)
validations = {
    "accuracy": accuracy,
    "f1_score": f1score_value,
    "cv_score": cv_score.mean()
}

df_random_forest_smote = pd.DataFrame(validations, index=["RandomForest SMOTE"])
print(df_random_forest_smote)