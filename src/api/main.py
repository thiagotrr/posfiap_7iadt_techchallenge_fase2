from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Query
from typing import Optional
import pandas as pd

from .schemas import PacienteHepaticoRequest, PredicaoResponse, TreinamentoModeloResponse, ModeloIA
from .carrega_modelo import carrega_modelo
from llm import gerar_consideracoes_clinicas

ml_items = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        artefatos = carrega_modelo()
        ml_items["model"] = artefatos["model"]
        ml_items["scaler"] = artefatos["scaler"]
        ml_items["features"] = list(artefatos["scaler"].feature_names_in_)
    except Exception as e:
        print(f"Erro ao carregar modelo: {e}")
    yield
    ml_items.clear()

app = FastAPI(
    title="API de Predição de Doença Hepática",
    description="FIAP - Pós-Tech IA para Devs - Tech Challenge Fase 2",
    version="1.0.0",
    lifespan=lifespan
)


def obter_predicao(paciente: PacienteHepaticoRequest) -> str:
    if "model" not in ml_items:
        raise HTTPException(status_code=503, detail="Modelo não carregado ou indisponível.")
        
    model = ml_items["model"]
    scaler = ml_items["scaler"]
    features = ml_items["features"]

    df_request_data = pd.DataFrame([paciente.model_dump()])
    
    # Carrega features
    df_model_features = df_request_data[features]

    # Aplica normalização
    X_input = scaler.transform(df_model_features)

    # Faz predição
    predicao = model.predict(X_input)

    resultado = "Potencial paciente." if predicao[0] == 1 else "Não é paciente."
    return resultado

@app.post("/predicao", 
          tags=["Modelo de Predição"],
          response_model=PredicaoResponse, 
          summary="Prediz se o paciente tem doença hepática (resposta crua do ML)"
        )
def predicao_paciente(paciente: PacienteHepaticoRequest):
    resultado = obter_predicao(paciente)
    return PredicaoResponse(
        resultado=resultado,
        consideracoes=""
    )

@app.post("/predicao-llm", 
          tags=["Modelo de Predição"],
          response_model=PredicaoResponse, 
          summary="Prediz se o paciente tem doença hepática com considerações clínicas geradas por LLM"
        )
def predicao_paciente_llm(
    requisicao: PacienteHepaticoRequest,
    modelo_ia: ModeloIA = Query(
        default=ModeloIA.GOOGLE_GEMINI_FLASH,
    )
):
    resultado = obter_predicao(requisicao)
    modelo_ia_str = modelo_ia.value
    
    try:
        consideracoes = gerar_consideracoes_clinicas(
            dados_paciente=requisicao.model_dump(),
            resultado_predicao=resultado,
            modelo_ia=modelo_ia_str
        )
        
        if consideracoes is None or "ERRO:" in consideracoes.upper():
            raise HTTPException(
                status_code=500,
                detail=f"Erro ao gerar considerações clínicas com modelo {modelo_ia_str}"
            )
    except HTTPException:
        raise
    except ValueError as e:
        raise HTTPException(
            status_code=400,
            detail=str(e)
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Erro ao processar requisição: {str(e)}"
        )
    
    return PredicaoResponse(
        resultado=resultado,
        consideracoes=consideracoes or ""
    )

@app.post("/treinar_modelo",
          tags=["Treinamento de Modelo"],
          summary="Treina um novo modelo de RandomForest com SMOTE e salva o modelo treinado"
        )
def treinar_novo_modelo():
    from ml.treinamento_modelo import treinar_modelo
    from ml.ferramentas_modelo import salvar_modelo

    modelo_treinado, scaler_treinado, df_metricas = treinar_modelo()
    # Erro se o treinamento não retornou um modelo
    if modelo_treinado is None:
        raise HTTPException(status_code=503, detail="Treinamento falhou: modelo não foi gerado.")

    caminho_salvo = salvar_modelo(modelo_treinado, scaler=scaler_treinado)
    if caminho_salvo is None:
        raise HTTPException(status_code=503, detail="Erro ao salvar modelo treinado.")

    hiperparametros = None
    if hasattr(modelo_treinado, "get_params"):
        hiperparametros = modelo_treinado.get_params()
    elif hasattr(modelo_treinado, "best_params_"):
        hiperparametros = modelo_treinado.best_params_
    elif hasattr(modelo_treinado, "best_estimator_") and hasattr(modelo_treinado.best_estimator_, "get_params"):
        hiperparametros = modelo_treinado.best_estimator_.get_params()
    else:
        hiperparametros = {}

    return TreinamentoModeloResponse(
        mensagem="Modelo treinado e salvo com sucesso.",
        caminho_modelo=caminho_salvo,
        metricas_validacao=df_metricas.to_dict(orient="records"),
        hiperparametros=hiperparametros
    )
@app.post("/otimizar_modelo", 
          tags=["Treinamento de Modelo"],
          summary="Otimiza hiperparâmetros do modelo RandomForest usando algoritmo genético",
          include_in_schema=False
          )
def otimizar_modelo():
    from ml.otimizador_modelo import otmgen_rdmforest_hepatico
    
    camminho_salvo = None
    df_metricas = None

    return TreinamentoModeloResponse(
        mensagem="otimização concluída com sucesso.",
        caminho_modelo=camminho_salvo, 
        metricas_validacao=df_metricas.to_dict(orient="records"),  # Pode ser preenchido com métricas reais se disponível
        hiperparametros={}  # Pode ser preenchido com os melhores hiperparâmetros encontrados
    )