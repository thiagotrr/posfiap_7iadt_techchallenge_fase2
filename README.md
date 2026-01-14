# posfiap_7iadt_techchallenge_fase2
# Tech Challenge Fase 2 - Pós Tech IA para Devs (Turma 7)

## Projeto Fase 2: Otimização de Modelos de Diagnóstico
O hospital precisa melhorar a precisão e eficiência dos modelos de diagnóstico desenvolvidos no Fase 1. O desafio é utilizar algoritmos genéticos para otimizar os hiperparâmetros desses modelos, além de  incorporar capacidades iniciais de processamento de linguagem natural por meio de LLMs para melhorar a interpretabilidade dos resultados para os profissionais de saúde.

## Instalação e execução
1. Fazer o clone do repositório;
2. Instalar os pacotes a partir de requirements.txt, conforme exemplo:
    ```bash
    pip install -r requirements.txt
    ```
3. A partir da raiz do projeto, executar:
    ```bash
    python main.py
     ```
4. No browser, acessar:
- Swagger: http://localhost:8181/docs
- OpenAPI: http://localhost:8181/openapi.json

## Recap Fase 1
### Dataset: Indian Liver Patient Dataset
A morte por cirrose hepática continua a aumentar, devido ao aumento nas taxas de consumo de álcool, infecções crônicas por hepatite e doenças hepáticas relacionadas à obesidade. Apesar da alta mortalidade dessa doença, as doenças do fígado não afetam todas as subpopulações de forma igual. A detecção precoce da patologia é determinante para o desfecho dos pacientes, mas as pacientes do sexo feminino parecem ser marginalizadas quando se trata do diagnóstico precoce de doenças hepáticas.

O conjunto de dados é composto por 584 registros de pacientes coletados no nordeste de Andhra Pradesh, na Índia. A tarefa de predição consiste em determinar se um paciente sofre de doença hepática com base em informações sobre diversos marcadores bioquímicos, incluindo albumina e outras enzimas necessárias para o metabolismo.

### Modelo Final: Random Forest com SMOTE

**Link notebook:** https://colab.research.google.com/drive/1hcM9gq6GKSIyd4yXzhrtj6E1O54fqEPE?usp=sharing

Considerando a finalidade acadêmica, e sabedores da sensibilidade em modelos voltados para área de saúde, tentamos duas últimas ações:
    - Aplicar o SMOTE com objetivo de mitigar o desbalanceamento das classes, conforme apresentado no material.
    - Utilizar o StratifiedKFold, que performa melhor com dados binários. Este, oriundo de pesquisa, que fez muito sentido já que aplicamos o stratifed no início do dataset.
Os hiperparâmetros refinados serão mantidos no treinamento do modelo.

### Conclusões Fase 1
Os resultados aplicando o SMOTE mostraram-se muito mais assertivos nas validações do modelo. O "RandomForest SMOTE" apresentou:
    - Capacidade de generalização
    - Melhor detecção da classe positiva
    - Maior acurácia no teste real

O modelo agora é excelente em identificar 'Não pacientes'. Dos 33 reais, ele acertou 30 (Verdadeiros Negativos) e errou apenas 3 (Falsos Positivos).

No entanto, o custo dessa melhoria foi que o modelo agora erra mais na classe 'Paciente'. Dos 83 reais, ele acertou 60 (Verdadeiros Positivos) e errou 23 (Falsos Negativos).

Em caso de evolução futura deste modelo, uma nova estratégia seria revalidar os hiperparâmetros.


| Modelo           |accuracy | f1_score | cv_score |
|------------------|---------|----------|----------|
|RandomForest SMOTE| 0.801724|  0.843537|  0.734239|

Com base na tabela dos resultados finais, o modelo RandomForest com SMOTE apresenta o melhor desempenho geral. Ele alcançou a maior acurácia (0.8017) e o maior cv_score (0.734) entre todos os modelos, além de um f1_score alto (0.843). O uso do SMOTE provavelmente ajudou a lidar com desequilíbrio da quantidade de pacientes e não pacientes, permitindo que o modelo generalizasse melhor sem favorecer apenas a classe majoritária (pacientes).