from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
import pandas as pd

from .schemas import PacienteHepaticoRequest, PredicaoResponse
from .carrega_modelo import carrega_modelo

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

@app.post("/predicao", 
          tags=["Modelo de Predição"],
          response_model=PredicaoResponse, 
          summary="Prediz se o paciente tem doença hepática"
        )
def predicao_paciente(paciente: PacienteHepaticoRequest):
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
    return PredicaoResponse(
        resultado=resultado,
        consideracoes= ("") ## LLM entra aqui para gerar considerações clínicas
    )