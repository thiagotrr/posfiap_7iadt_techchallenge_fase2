from pydantic import BaseModel, Field

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
                "consideracoes": "Considerações clínicas relevantes"
                }
        }