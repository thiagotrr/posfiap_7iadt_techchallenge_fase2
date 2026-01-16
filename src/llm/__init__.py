import os
from pathlib import Path
from typing import Dict, Any, Optional
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent.parent
env_path = BASE_DIR / ".env"
load_dotenv(dotenv_path=env_path)

def _gerar_mensagem_padrao(dados_paciente: Dict[str, Any], resultado_predicao: str, mensagem_extra: str = "") -> str:
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
"""
    
    if mensagem_extra:
        consideracoes += f"\n{mensagem_extra}\n"
    
    consideracoes += """
IMPORTANTE: Esta é uma predição assistida por IA baseada em modelos de aprendizado de máquina e não substitui a avaliação médica profissional. Recomenda-se consultar um médico especialista para interpretação adequada dos resultados e orientações sobre próximos passos.
"""
    return consideracoes.strip()

def _obter_configuracao_llm(modelo_ia: str) -> Dict[str, Any]:
    configs = {
        "openai/gpt-4o-mini": {
            "model": "gpt-4o-mini",
            "api_key": os.getenv("OPENAI_API_KEY"),
            "base_url": "https://api.openai.com/v1",
            "provider": "openai"
        },
        "groq/llama-3.3-70b-versatile": {
            "model": "llama-3.3-70b-versatile",
            "api_key": os.getenv("GROQ_API_KEY"),
            "base_url": "https://api.groq.com/openai/v1",
            "provider": "groq"
        },
        "google/gemini-3-flash-preview": {
            "model": "gemini-3-flash-preview",
            "api_key": os.getenv("GOOGLE_API_KEY"),
            "base_url": "https://generativelanguage.googleapis.com/v1",
            "provider": "google"
        },
    }
    return configs.get(modelo_ia, {})


def _criar_prompt(dados_paciente: Dict[str, Any], resultado_predicao: str) -> str:
    """Cria o prompt para o LLM"""
    return f"""Você é um assistente médico especializado em doenças hepáticas. 

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

def _configurar_llm(config: Dict[str, Any], provider: str) -> Any:
    """Configura e retorna o LLM usando CrewAI"""
    from crewai import LLM
    
    model_map = {
        "groq": f"groq/{config['model']}",
        "openai": f"openai/{config['model']}",
        "google": f"gemini/{config['model']}"
    }
    model_name = model_map.get(provider, config["model"])
    
    env_map = {
        "groq": "GROQ_API_KEY",
        "openai": "OPENAI_API_KEY",
        "google": "GEMINI_API_KEY"
    }
    if provider in env_map and config.get("api_key"):
        os.environ[env_map[provider]] = config.get("api_key")
    
    return LLM(model=model_name, temperature=0.7)

def _executar_crew_ai(prompt: str, llm: Any) -> str:
    from crewai import Agent, Task, Crew
    
    agente = Agent(
        role="Especialista em Doenças Hepáticas",
        goal="Fornecer análises clínicas precisas e compreensíveis sobre doenças hepáticas baseadas em dados laboratoriais",
        backstory="Você é um médico experiente especializado em hepatologia, com anos de experiência "
                 "em análise de exames laboratoriais relacionados a doenças do fígado. "
                 "Você é conhecido por suas análises claras e objetivas, sempre priorizando o bem-estar do paciente.",
        llm=llm,
        verbose=False,
        allow_delegation=False
    )
    
    task = Task(
        description=prompt,
        agent=agente,
        expected_output="Um texto claro e conciso (máximo 200 palavras) com considerações clínicas sobre a predição de doença hepática, incluindo interpretação do resultado, análise dos exames e recomendações"
    )
    
    crew = Crew(agents=[agente], tasks=[task], verbose=False)
    resultado = crew.kickoff()
    
    if isinstance(resultado, str):
        return resultado
    if hasattr(resultado, 'raw'):
        return resultado.raw
    if hasattr(resultado, 'output'):
        return resultado.output
    if hasattr(resultado, 'tasks_output'):
        tasks_output = resultado.tasks_output
        if isinstance(tasks_output, list) and len(tasks_output) > 0:
            task_result = tasks_output[0]
            if hasattr(task_result, 'raw'):
                return task_result.raw
            if hasattr(task_result, 'output'):
                return task_result.output
            return str(task_result)
        return str(tasks_output)
    return str(resultado)

def gerar_consideracoes_clinicas(dados_paciente: Dict[str, Any], resultado_predicao: str, modelo_ia: str = "default") -> str:
    if modelo_ia == "default" or not modelo_ia:
        return _gerar_mensagem_padrao(dados_paciente, resultado_predicao)
    
    config = _obter_configuracao_llm(modelo_ia)
    
    if not config:
        raise ValueError(f"Modelo de IA '{modelo_ia}' não está configurado ou não é suportado.")
    
    try:
        prompt = _criar_prompt(dados_paciente, resultado_predicao)
        provider = config.get("provider", modelo_ia.split("/")[0] if "/" in modelo_ia else "")
        llm = _configurar_llm(config, provider)
        texto_resultado = _executar_crew_ai(prompt, llm)
        
        if texto_resultado and isinstance(texto_resultado, str) and texto_resultado.strip():
            return texto_resultado.strip()
        return _gerar_mensagem_padrao(dados_paciente, resultado_predicao)
            
    except ImportError:
        print(f"Warning: CrewAI não disponível")
        return _gerar_mensagem_padrao(dados_paciente, resultado_predicao)
    except Exception as e:
        print(f"Erro ao usar LLM {modelo_ia}: {str(e)}")
        return _gerar_mensagem_padrao(dados_paciente, resultado_predicao)

