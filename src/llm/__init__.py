import os
from pathlib import Path
from typing import Dict, Any, Optional
from dotenv import load_dotenv

# utilizacao de arquivo .env para armazenar a API key da Google GenAI
BASE_DIR = Path(__file__).resolve().parent.parent.parent
env_path = BASE_DIR / ".env"
load_dotenv(dotenv_path=env_path)

_client: Optional[Any] = None

def _get_client():
    global _client
    if _client is None:
        from google import genai
        api_key = os.getenv("GOOGLE_API_KEY")
        if api_key:
            _client = genai.Client(api_key=api_key)
        else:
            _client = None
    return _client

def gerar_consideracoes_clinicas(dados_paciente: Dict[str, Any], resultado_predicao: str) -> str:
    """
    Gera considerações clínicas baseadas nos dados do paciente e na predição do modelo.
    
    Args:
        dados_paciente: Dicionário com os dados do paciente
        resultado_predicao: Resultado da predição ("Potencial paciente." ou "Não é paciente.")
    
    Returns:
        Considerações clínicas geradas pela LLM prontas para serem exibidas em uma interface de usuário através de uma página HTML.
    """
    prompt = f"""Você é um assistente médico especializado em doenças hepáticas. 
    
    Com base nos seguintes dados do paciente e no resultado da predição do modelo de aprendizado de máquina, 
    forneça considerações clínicas relevantes em linguagem clara e acessível.

    Dados do paciente:
    - Idade: {dados_paciente.get('age')} anos
    - Bilirrubina total: {dados_paciente.get('tot_bilirubin')} mg/dL
    - Bilirrubina direta: {dados_paciente.get('direct_bilirubin')} mg/dL
    - Proteínas totais: {dados_paciente.get('tot_proteins')} g/dL
    - Albumina: {dados_paciente.get('albumin')} g/dL
    - Relação Albumina/Globulina: {dados_paciente.get('ag_ratio')}
    - SGPT (ALT): {dados_paciente.get('sgpt')} U/L
    - SGOT (AST): {dados_paciente.get('sgot')} U/L
    - Fosfatase alcalina: {dados_paciente.get('alkphos')} U/L

    Resultado da predição: {resultado_predicao}

    Por favor, forneça:
    1. Uma interpretação clara do resultado da predição
    2. Análise breve dos valores dos exames em relação aos valores de referência
    3. Recomendações gerais de próximos passos (se aplicável)

    IMPORTANTE: Mantenha a resposta concisa (máximo 200 palavras) e sempre enfatize que esta é uma predição assistida por IA e não substitui avaliação médica profissional.
    """

    try:
        client = _get_client()
        if client is None:
            return _gerar_mensagem_padrao(dados_paciente, resultado_predicao)
        
        response = client.models.generate_content(
            model="gemini-3-flash-preview",
            contents=prompt,
        )
        texto_resposta = response.text if hasattr(response, 'text') and response.text else ""
        if not texto_resposta or texto_resposta.strip() == "":
            return _gerar_mensagem_padrao(dados_paciente, resultado_predicao)
        return texto_resposta
    except Exception as e:
        return _gerar_mensagem_padrao(dados_paciente, resultado_predicao)

def _gerar_mensagem_padrao(dados_paciente: Dict[str, Any], resultado_predicao: str) -> str:
    """
    Gera uma mensagem padrão baseada nos dados do paciente quando não há API key configurada.
    """
    resultado_texto = "O modelo de aprendizado de máquina indicou que este paciente apresenta sinais compatíveis com doença hepática." if "Potencial" in resultado_predicao else "O modelo de aprendizado de máquina indicou que este paciente não apresenta sinais compatíveis com doença hepática."
    
    consideracoes = f"""{resultado_texto}

    Análise dos exames:
    - Bilirrubina total: {dados_paciente.get('tot_bilirubin', 'N/A')} mg/dL (referência: 0,3-1,2 mg/dL)
    - Bilirrubina direta: {dados_paciente.get('direct_bilirubin', 'N/A')} mg/dL (referência: <0,3 mg/dL)
    - Proteínas totais: {dados_paciente.get('tot_proteins', 'N/A')} g/dL (referência: 6,0-8,3 g/dL)
    - Albumina: {dados_paciente.get('albumin', 'N/A')} g/dL (referência: 3,5-5,0 g/dL)
    - SGPT (ALT): {dados_paciente.get('sgpt', 'N/A')} U/L (referência: 7-56 U/L)
    - SGOT (AST): {dados_paciente.get('sgot', 'N/A')} U/L (referência: 10-40 U/L)
    - Fosfatase alcalina: {dados_paciente.get('alkphos', 'N/A')} U/L (referência: 44-147 U/L)

    IMPORTANTE: Esta é uma predição assistida por IA baseada em modelos de aprendizado de máquina e não substitui a avaliação médica profissional. Recomenda-se consultar um médico especialista para interpretação adequada dos resultados e orientações sobre próximos passos.
    """
    return consideracoes.strip()
