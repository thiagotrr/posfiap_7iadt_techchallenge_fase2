from pydantic import BaseModel, Field, ConfigDict
from typing import Optional

class PacienteHepaticoRequest(BaseModel):
    age: int = Field(..., description="Idade do paciente em anos", ge=0, le=100, example=39)
    tot_bilirubin: float = Field(..., description="Bilirrubina total", example=1.2)
    direct_bilirubin: float = Field(..., description="Bilirrubina direta", example=0.5)
    tot_proteins: float = Field(..., description="Proteínas totais", example=6.5)
    albumin: float = Field(..., description="Albumina", example=3.5)
    ag_ratio: float = Field(..., description="Relação Álbumina/Globulina", example=1.2)
    sgpt: float = Field(..., description="Transaminase Glutâmico-Pirúvica (TGP/ALT)", example=40.0)  
    sgot: float = Field(..., description="Transaminase Glutâmico-Oxalacética (TGO/AST)", example=40.0)
    alkphos: float = Field(..., description="Alcalina Fosfatase", example=210.0)

    class Config:
        json_schema_extra = {
            "example": {
                "age": 39,
                "tot_bilirubin": 1.2,
                "direct_bilirubin": 0.5,
                "tot_proteins": 6.5,
                "albumin": 3.5,
                "ag_ratio": 1.2,
                "sgpt": 40.0,
                "sgot": 40.0,
                "alkphos": 210.0
            }
        }

class PredicaoResponse(BaseModel):
    resultado: str = Field(..., 
                           description="Resultado da predição: 'Potencial paciente' ou 'Não é paciente'", 
                           example="Potencial paciente."
                          )
    consideracoes: str = Field(..., 
                              description="Considerações adicionais sobre a predição, considerando o contexto clínico.", 
                              example="O paciente apresenta níveis elevados de bilirrubina e enzimas hepáticas, indicando possível comprometimento hepático."
                              )  

    class Config:
        json_schema_extra = {
            "example": {
                "resultado": "Potencial paciente.",
                "consideracoes": "Considerações clínicas relevantes descritas pela LLM"
                }
        }

class PredicaoLLMRequest(BaseModel):
    """Schema para requisição de predição com LLM"""
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "paciente": {
                    "age": 39,
                    "tot_bilirubin": 1.2,
                    "direct_bilirubin": 0.5,
                    "tot_proteins": 6.5,
                    "albumin": 3.5,
                    "ag_ratio": 1.2,
                    "sgpt": 40.0,
                    "sgot": 40.0,
                    "alkphos": 210.0
                },
            }
        }
    )
    
    paciente: PacienteHepaticoRequest = Field(..., description="Dados do paciente")

class TreinamentoModeloResponse(BaseModel):
    mensagem: str = Field(..., description="Mensagem indicando o status do treinamento do modelo", examples=["Modelo treinado", "Modelo treinado com otimização"])
    caminho_modelo: str = Field(..., description="Caminho onde o modelo treinado foi salvo", example="/modelos/modelo_treinado_v1.joblib")
    id: str = Field(..., description="Identificador hexadecimal do experimento/treinamento", example="a1b2c3")
    metricas_validacao: dict = Field(..., description="Métricas de validação do modelo treinado", example={"acurácia": 0.85, "f1_score": 0.83, "recall": 0.82, "cv_score": 0.84})
    hiperparametros: dict = Field(..., description="Hiperparâmetros do modelo treinado", example={"n_estimators": 100, "max_depth": 5, "criterion": "gini"})

    class Config:
        json_schema_extra = {
            "example": {
                "mensagem": "Modelo treinado com sucesso.",
                "caminho_modelo": "/modelos/modelo_treinado_v1.joblib",
                "id": "a1b2c3",
                "metricas_validacao": {
                    "acurácia": 0.85,
                    "f1_score": 0.83,
                    "recall": 0.82,
                    "cv_score": 0.84
                },
                "hiperparametros": {
                    "n_estimators": 100,
                    "max_depth": 5,
                    "criterion": "gini"
                }
            }
        }


class OtimizacaoRequest(BaseModel):
    geracoes: int = Field(10, description="Número de gerações para o algoritmo genético", example=10, ge=1)
    tamanho_populacao: int = Field(12, description="Tamanho da população por geração", example=12, ge=2)
    k_torneio: int = Field(3, description="Tamanho do torneio para seleção dos pais", example=3, ge=2)
    prob_cruzamento: float = Field(0.9, description="Probabilidade de cruzamento", example=0.9, ge=0.0, le=1.0)
    prob_mutacao: float = Field(0.3, description="Probabilidade de mutação", example=0.3, ge=0.0, le=1.0)
    cv: int = Field(3, description="Número de folds para cross-validation", example=3, ge=2)
    semente: Optional[int] = Field(None, description="Semente aleatória (opcional)", example=42)
    caminho_log: Optional[str] = Field(None, description="Nome do arquivo de log (opcional). Se não informado, será gerado automaticamente com timestamp e id")

    class Config:
        json_schema_extra = {
            "example": {
                "geracoes": 3,
                "tamanho_populacao": 6,
                "k_torneio": 3,
                "prob_cruzamento": 0.9,
                "prob_mutacao": 0.3,
                "cv": 3,
                "semente": 42,
                "caminho_log": "otimizacao.log"
            }
        }


class FileDownloadResponse(BaseModel):
    filename: str = Field(..., description="Nome do arquivo retornado", example="rdm_forest_smote_v0_a1b2c3.joblib")
    content_base64: str = Field(..., description="Conteúdo do arquivo codificado em base64", example="UEsDB...==")

    class Config:
        json_schema_extra = {
            "example": {
                "filename": "rdm_forest_smote_v0_a1b2c3.joblib",
                "content_base64": "UEsDB...=="
            }
        }