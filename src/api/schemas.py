from pydantic import BaseModel, Field

class PacienteHepático(BaseModel):
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
    schema_extra = {
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