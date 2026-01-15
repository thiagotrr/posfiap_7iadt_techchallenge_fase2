import random
from typing import List, Dict, Any, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import make_scorer, recall_score
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import MinMaxScaler
from .ferramentas_modelo import carregar_dataset, salvar_modelo

# Espaços de busca (valores possíveis)
RANGE_N_ESTIMATORS = (10, 300)
OPCOES_MAX_DEPTH = [None, 3, 5, 7, 9, 12, 15]
RANGE_MIN_SAMPLES_SPLIT = (2, 20)
RANGE_MIN_SAMPLES_LEAF = (1, 10)
OPCOES_CRITERION = ["gini", "entropy"]

_ORDEM_GENES = ["n_estimators", "max_depth", "min_samples_split", "min_samples_leaf", "criterion"]


def gerar_individuo_aleatorio(aleatorio: random.Random) -> Dict[str, Any]:
    """Geração de indivíduo"""
    return {
        "n_estimators": aleatorio.randint(*RANGE_N_ESTIMATORS),
        "max_depth": aleatorio.choice(OPCOES_MAX_DEPTH),
        "min_samples_split": aleatorio.randint(*RANGE_MIN_SAMPLES_SPLIT),
        "min_samples_leaf": aleatorio.randint(*RANGE_MIN_SAMPLES_LEAF),
        "criterion": aleatorio.choice(OPCOES_CRITERION),
    }


def dict_para_lista(ind: Dict[str, Any]) -> List[Any]:
    """Conversão dict -> lista de genes"""
    return [ind[k] for k in _ORDEM_GENES]


def lista_para_dict(lista: List[Any]) -> Dict[str, Any]:
    """Conversão lista -> dict de hiperparâmetros"""
    return {k: lista[i] for i, k in enumerate(_ORDEM_GENES)}


def cruzamento_order(parent_a: List[Any], parent_b: List[Any], aleatorio: random.Random) -> List[Any]:
    """Cruzamento Order Crossover"""
    tamanho = len(parent_a)
    i, j = sorted(aleatorio.sample(range(tamanho), 2))
    filho = [None] * tamanho
    # copia segmento de A
    filho[i : j + 1] = parent_a[i : j + 1]
    # preenche o resto com a ordem de B, ignorando valores já copiados
    ponteiro = (j + 1) % tamanho
    inserir = (j + 1) % tamanho
    while None in filho:
        val = parent_b[ponteiro]
        if val not in filho:
            filho[inserir] = val
            inserir = (inserir + 1) % tamanho
        ponteiro = (ponteiro + 1) % tamanho
    return filho


def mutacao_probabilistica(lista: List[Any], prob_mutacao: float, aleatorio: random.Random) -> List[Any]:
    """Mutação probabilística"""
    nova = lista.copy()
    # Para cada posição, com probabilidade altera para um valor aleatório válido
    for idx, gene in enumerate(nova):
        if aleatorio.random() < prob_mutacao:
            chave = _ORDEM_GENES[idx]
            if chave == "n_estimators":
                nova[idx] = aleatorio.randint(*RANGE_N_ESTIMATORS)
            elif chave == "max_depth":
                nova[idx] = aleatorio.choice(OPCOES_MAX_DEPTH)
            elif chave == "min_samples_split":
                nova[idx] = aleatorio.randint(*RANGE_MIN_SAMPLES_SPLIT)
            elif chave == "min_samples_leaf":
                nova[idx] = aleatorio.randint(*RANGE_MIN_SAMPLES_LEAF)
            elif chave == "criterion":
                nova[idx] = aleatorio.choice(OPCOES_CRITERION)
    return nova


def selecao_torneio(pop: List[Tuple[Dict[str, Any], float]], k: int, aleatorio: random.Random) -> Dict[str, Any]:
    """Seleção por torneio"""
    candidatos = aleatorio.sample(pop, k)
    candidatos.sort(key=lambda x: x[1], reverse=True)
    return candidatos[0][0].copy()


def avaliar_individuo(param_dict: Dict[str, Any], X, y, cv: int, semente: Optional[int]) -> float:
    """Avaliar indivíduo (fitness = recall CV com SMOTE)"""
    sm = SMOTE(random_state=semente)
    Xr, yr = sm.fit_resample(X, y)
    clf = RandomForestClassifier(
        n_estimators=int(param_dict["n_estimators"]),
        max_depth=param_dict["max_depth"],
        min_samples_split=int(param_dict["min_samples_split"]),
        min_samples_leaf=int(param_dict["min_samples_leaf"]),
        criterion=param_dict["criterion"],
        random_state=semente,
        n_jobs=1,
    )
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=semente)
    scorer = make_scorer(recall_score, greater_is_better=True)
    scores = cross_val_score(clf, Xr, yr, cv=skf, scoring=scorer, n_jobs=1)
    return float(np.mean(scores))


# dataset loading and saving functionality moved to `ferramentas_modelo.py`


def otmgen_rdmforest_hepatico(
                                geracoes: int = 10,
                                tamanho_populacao: int = 12,
                                k_torneio: int = 3,
                                prob_cruzamento: float = 0.9,
                                prob_mutacao: float = 0.3,
                                cv: int = 3,
                                semente: Optional[int] = 42,
                                caminho_salvar: Optional[str] = None,
                            ) -> Tuple[Dict[str, Any], float, RandomForestClassifier]:
    """Otimização genética para o algoritmo RandomForestClassifier em dados hepáticos"""
    rnd = random.Random(semente)

    # carrega dataset internamente
    X, y, scaler = carregar_dataset()

    # inicializa população
    populacao = [gerar_individuo_aleatorio(rnd) for _ in range(tamanho_populacao)]

    # avalia inicial
    avaliacoes: List[Tuple[Dict[str, Any], float]] = []
    for ind in populacao:
        score = avaliar_individuo(ind, X, y, cv=cv, semente=semente)
        avaliacoes.append((ind, score))

    avaliacoes.sort(key=lambda x: x[1], reverse=True)
    melhor_ind, melhor_score = avaliacoes[0][0].copy(), avaliacoes[0][1]

    for g in range(geracoes):
        nova_pop: List[Dict[str, Any]] = []
        # elitismo simples (mantém 1)
        avaliacoes.sort(key=lambda x: x[1], reverse=True)
        nova_pop.append(avaliacoes[0][0].copy())

        # gera filhos
        while len(nova_pop) < tamanho_populacao:
            pai_a = selecao_torneio(avaliacoes, k_torneio, rnd)
            pai_b = selecao_torneio(avaliacoes, k_torneio, rnd)
            lista_a = dict_para_lista(pai_a)
            lista_b = dict_para_lista(pai_b)
            if rnd.random() < prob_cruzamento:
                filho_lista = cruzamento_order(lista_a, lista_b, rnd)
            else:
                filho_lista = lista_a.copy()
            filho_lista = mutacao_probabilistica(filho_lista, prob_mutacao, rnd)
            filho = lista_para_dict(filho_lista)
            nova_pop.append(filho)

        # avalia nova pop
        avaliacoes = []
        for ind in nova_pop:
            score = avaliar_individuo(ind, X, y, cv=cv, semente=semente)
            avaliacoes.append((ind, score))

        avaliacoes.sort(key=lambda x: x[1], reverse=True)
        if avaliacoes[0][1] > melhor_score:
            melhor_ind, melhor_score = avaliacoes[0][0].copy(), avaliacoes[0][1]

    # treina modelo final com SMOTE usando melhores hiperparâmetros
    sm = SMOTE(random_state=semente)
    Xr, yr = sm.fit_resample(X, y)
    melhor_modelo = RandomForestClassifier(
        n_estimators=int(melhor_ind["n_estimators"]),
        max_depth=melhor_ind["max_depth"],
        min_samples_split=int(melhor_ind["min_samples_split"]),
        min_samples_leaf=int(melhor_ind["min_samples_leaf"]),
        criterion=melhor_ind["criterion"],
        random_state=semente,
        n_jobs=-1,
    )
    melhor_modelo.fit(Xr, yr)

    if caminho_salvar:
        # Use unified salvar_modelo which preserves versioning and accepts explicit path and params
        salvar_modelo(melhor_modelo, scaler=scaler, params=melhor_ind, caminho_salvar=caminho_salvar)

    return melhor_ind, float(melhor_score), melhor_modelo


# Exemplo de uso rápido
if __name__ == "__main__":
    import os
    import kagglehub as kh
    from sklearn.preprocessing import MinMaxScaler

    melhores, score, modelo = otmgen_rdmforest_hepatico(geracoes=3, tamanho_populacao=6, cv=3, semente=42)
    print("Melhores:", melhores)
    print("Recall (CV):", score)
