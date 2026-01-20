import logging
from typing import Any, Dict, Optional, Tuple

import random
import numpy as np
import pygad
import os
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE

from .ferramentas_modelo import carregar_dataset, salvar_modelo

# Resgate explícito dos genes (copiado de `otimizador_modelo`) - apenas a lista
ORDEM_DOS_GENES = [
    "n_estimators",
    "max_depth",
    "min_samples_split",
    "min_samples_leaf",
    "criterion",
]

# Valores padrão de entrada (copiados do módulo original)
PADRAO_ENTRADA = {
    "geracoes": 10,
    "tamanho_populacao": 12,
    "k_torneio": 3,
    "prob_cruzamento": 0.9,
    "prob_mutacao": 0.3,
    "cv": 3,
    "semente": 42,
    "caminho_salvar_modelo": None,
    "caminho_log": None,
}

# Espaços de busca modestos (não muito grandes) - usados no PyGAD
N_ESTIMATORS_CHOICES = list(range(50, 151, 10))
# representamos None como -1 internamente e converteremos depois
MAX_DEPTH_CHOICES = [-1, 5, 10]
MIN_SAMPLES_SPLIT_CHOICES = list(range(2, 11))
MIN_SAMPLES_LEAF_CHOICES = list(range(1, 5))
CRITERION_MAP = ["gini", "entropy"]
CRITERION_CHOICES = list(range(len(CRITERION_MAP)) )


def configurar_logger(nome: str = "otimiza_modelo", caminho_log: Optional[str] = None) -> logging.Logger:
    logger = logging.getLogger(nome)
    logger.setLevel(logging.INFO)
    if logger.handlers:
        return logger

    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    logger.addHandler(sh)

    if caminho_log:
        fh = logging.FileHandler(caminho_log, encoding="utf-8")
        fh.setFormatter(fmt)
        logger.addHandler(fh)

    return logger


def list_para_parametros(solucao: list) -> Dict[str, Any]:
    # solucao chega como lista de valores escolhidos do gene_space
    n_estimators = int(solucao[0])
    max_depth = int(solucao[1])
    if max_depth == -1:
        max_depth = None
    min_samples_split = int(solucao[2])
    min_samples_leaf = int(solucao[3])
    # criterion can be an index (int) chosen from CRITERION_CHOICES
    crit_idx = int(solucao[4])
    criterion = CRITERION_MAP[crit_idx]
    return {
        "n_estimators": n_estimators,
        "max_depth": max_depth,
        "min_samples_split": min_samples_split,
        "min_samples_leaf": min_samples_leaf,
        "criterion": criterion,
    }


def avaliar_solucao(solucao, dados):
    # função de fitness para PyGAD; maximizar recall
    X, y, cv, semente = dados
    params = list_para_parametros(solucao)
    sm = SMOTE(random_state=semente)
    Xr, yr = sm.fit_resample(X, y)
    clf = RandomForestClassifier(
        n_estimators=int(params["n_estimators"]),
        max_depth=params["max_depth"],
        min_samples_split=int(params["min_samples_split"]),
        min_samples_leaf=int(params["min_samples_leaf"]),
        criterion=params["criterion"],
        random_state=semente,
        n_jobs=1,
    )
    from sklearn.model_selection import cross_val_score, StratifiedKFold
    from sklearn.metrics import make_scorer, recall_score

    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=semente)
    scorer = make_scorer(recall_score, greater_is_better=True)
    scores = cross_val_score(clf, Xr, yr, cv=skf, scoring=scorer, n_jobs=1)
    return float(np.mean(scores))


def otimizar_random_forest(
                            geracoes: int = PADRAO_ENTRADA["geracoes"],
                            tamanho_populacao: int = PADRAO_ENTRADA["tamanho_populacao"],
                            k_torneio: int = PADRAO_ENTRADA["k_torneio"],
                            prob_cruzamento: float = PADRAO_ENTRADA["prob_cruzamento"],
                            prob_mutacao: float = PADRAO_ENTRADA["prob_mutacao"],
                            cv: int = PADRAO_ENTRADA["cv"],
                            semente: Optional[int] = PADRAO_ENTRADA["semente"],
                            caminho_salvar_modelo: Optional[str] = PADRAO_ENTRADA["caminho_salvar_modelo"],
                            caminho_log: Optional[str] = PADRAO_ENTRADA["caminho_log"],
                            id_hex: Optional[str] = None,
                        ) -> Tuple[Dict[str, Any], float, RandomForestClassifier]:
    # preparar pasta e arquivo de log
    logs_dir = os.path.join("src", "ml", "log")
    os.makedirs(logs_dir, exist_ok=True)

    # sempre gerar nome padrão com timestamp e id (quando presente) para evitar nomes incorretos
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    suff = f"_{id_hex}" if id_hex else ""
    caminho_log = os.path.join(logs_dir, f"otm_rdmfst_{ts}{suff}.log")

    # criar um logger único por execução para evitar reuso de handlers
    nome_logger = f"otimiza_modelo_{ts}{suff}"
    logger = configurar_logger(nome=nome_logger, caminho_log=caminho_log)
    logger.info("Iniciando otimização com PyGAD (Random Forest)")

    rnd = random.Random(semente)

    logger.info("Carregando dataset e scaler")
    X, y, scaler = carregar_dataset()

    logger.info("Espaço de busca: n_estimators=%s, max_depth=%s, min_samples_split=%s, min_samples_leaf=%s, criterion=%s",
                N_ESTIMATORS_CHOICES, MAX_DEPTH_CHOICES, MIN_SAMPLES_SPLIT_CHOICES, MIN_SAMPLES_LEAF_CHOICES, CRITERION_MAP)

    # montar gene_space para PyGAD
    gene_space = [
        N_ESTIMATORS_CHOICES,
        MAX_DEPTH_CHOICES,
        MIN_SAMPLES_SPLIT_CHOICES,
        MIN_SAMPLES_LEAF_CHOICES,
        CRITERION_CHOICES,
    ]

    dados = (X, y, cv, semente)

    def fitness_func(ga_instance, solucao, indice_solucao):
        return avaliar_solucao(solucao, dados)

    num_genes = len(gene_space)
    num_parents = max(2, tamanho_populacao // 2)
    logger.info("Config: gerações=%d, população=%d, pais_por_cruzamento=%d, cv=%d, semente=%s",
                geracoes, tamanho_populacao, num_parents, cv, str(semente))

    # callback por geração para logs mais densos
    def callback_geracao(ga_inst):
        try:
            sols = ga_inst.population
            fits = np.array(ga_inst.last_generation_fitness)
        except Exception:
            logger.info("Geração %d concluída (resumo indisponível)", ga_inst.generations_completed)
            return

        if fits.size == 0:
            logger.info("Geração %d concluída (sem fitness)", ga_inst.generations_completed)
            return

        melhor_idx = int(np.argmax(fits))
        melhor_fit = float(fits[melhor_idx])
        media = float(np.mean(fits))
        desv = float(np.std(fits))
        logger.info("Geração %d — melhor recall=%.4f, média=%.4f, std=%.4f", ga_inst.generations_completed, melhor_fit, media, desv)

        # logar top-3 soluções da geração
        ordenados = np.argsort(fits)[::-1]
        topk = min(3, ordenados.size)
        for r in range(topk):
            idx = int(ordenados[r])
            sol = sols[idx]
            params = list_para_parametros(sol.tolist() if hasattr(sol, 'tolist') else list(sol))
            logger.info(" G%d-R%d: %s -> recall=%.4f", ga_inst.generations_completed, r + 1, params, float(fits[idx]))


    ga = pygad.GA(
        num_generations=geracoes,
        sol_per_pop=tamanho_populacao,
        num_parents_mating=num_parents,
        num_genes=num_genes,
        gene_space=gene_space,
        fitness_func=fitness_func,
        parent_selection_type="tournament",
        K_tournament=k_torneio,
        crossover_type="single_point",
        mutation_type="random",
        mutation_probability=prob_mutacao,
        crossover_probability=prob_cruzamento,
        random_seed=semente,
        suppress_warnings=True,
        on_generation=callback_geracao,
    )

    ga.run()
    melhor_solucao, melhor_fitness, _ = ga.best_solution()
    melhor_params = list_para_parametros(melhor_solucao)

    logger.info("Melhor solução encontrada: %s (recall=%.4f)", melhor_params, float(melhor_fitness))

    # treinar modelo final com SMOTE
    sm = SMOTE(random_state=semente)
    Xr, yr = sm.fit_resample(X, y)
    modelo = RandomForestClassifier(
        n_estimators=int(melhor_params["n_estimators"]),
        max_depth=melhor_params["max_depth"],
        min_samples_split=int(melhor_params["min_samples_split"]),
        min_samples_leaf=int(melhor_params["min_samples_leaf"]),
        criterion=melhor_params["criterion"],
        random_state=semente,
        n_jobs=-1,
    )
    modelo.fit(Xr, yr)

    if caminho_salvar_modelo:
        logger.info("Salvando modelo em %s", caminho_salvar_modelo)
        salvar_modelo(modelo, scaler=scaler, params=melhor_params, caminho_salvar=caminho_salvar_modelo, id_hex=id_hex)

    logger.info("Otimização finalizada. Melhor recall=%.4f", float(melhor_fitness))

    # Garantir que handlers de arquivo sejam fechados e liberem o arquivo de log
    for h in list(logger.handlers):
        try:
            if isinstance(h, logging.FileHandler):
                try:
                    h.flush()
                except Exception:
                    pass
                try:
                    h.close()
                except Exception:
                    pass
                logger.removeHandler(h)
        except Exception:
            # não falhar o processo de retorno por causa de problemas com handlers
            continue

    return melhor_params, float(melhor_fitness), modelo


if __name__ == "__main__":
    melhores, score, modelo = otimizar_random_forest(geracoes=3, tamanho_populacao=6, cv=3, semente=42, caminho_log="otimizacao.log")
    print("Melhores:", melhores)
    print("Recall (CV):", score)
