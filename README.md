# 🏆 Tech Challenge Fase 2 - Pós Tech IA para Devs (posfiap_7iadt_techchallenge_fase2)

## 🎯 Projeto Fase 2: Otimização de Modelos de Diagnóstico
O hospital precisa melhorar a precisão e eficiência dos modelos de diagnóstico desenvolvidos na Fase 1. O desafio é utilizar algoritmos genéticos para otimizar os hiperparâmetros desses modelos, além de incorporar capacidades iniciais de processamento de linguagem natural por meio de LLMs para melhorar a interpretabilidade dos resultados para os profissionais de saúde.

## 🛠️ Instalação e execução
1.  **Clone o repositório:**
    ```bash
    git clone <URL_DO_REPOSITORIO>
    ```
2.  **Instale as dependências** a partir do `requirements.txt`:
    ```bash
    pip install -r requirements.txt
    ```
3.  **Execute a aplicação** a partir da raiz do projeto:
    ```bash
    python main.py
    ```
4.  **Acesse a API** no seu navegador:
    -   **Swagger UI**: [http://localhost:8181/docs](http://localhost:8181/docs) 接口
    -   **OpenAPI JSON**: [http://localhost:8181/openapi.json](http://localhost:8181/openapi.json) 📄

## ✨ Features Projeto Fase 2

Este projeto expande as capacidades do modelo de diagnóstico da Fase 1, introduzindo novas funcionalidades para treinamento, otimização e rastreabilidade.

### 🚀 Treinamento e Otimização do Modelo

- **Treinamento via API**: Execute o treinamento de um novo modelo `RandomForest` com `SMOTE` a qualquer momento através do endpoint `/treinar_modelo`.
- **Otimização com Algoritmo Genético**: Utilize o endpoint `/otimizar_modelo` para otimizar os hiperparâmetros do modelo. O processo utiliza um algoritmo genético para encontrar a melhor combinação de parâmetros, maximizando a performance.

### 🔗 Identificador Único e Rastreabilidade

- Cada processo de **treinamento** ou **otimização** gera um **identificador único hexadecimal**.
- Esse `id` cria um vínculo direto e inequívoco entre o **modelo treinado** (arquivo `.joblib`) e seu respectivo **arquivo de log**, garantindo total rastreabilidade dos artefatos gerados.

### ⬇️ Download de Artefatos

- **Acesso direto aos modelos e logs**: Após o treinamento ou otimização, utilize os endpoints `/modelo/{id_hex}` e `/log/{id_hex}` para baixar os arquivos gerados.
- Facilita a análise de performance, o reuso de modelos e a auditoria do processo.

## ⏪ Recap Fase 1
### 📊 Dataset: Indian Liver Patient Dataset
A morte por cirrose hepática continua a aumentar, devido ao aumento nas taxas de consumo de álcool, infecções crônicas por hepatite e doenças hepáticas relacionadas à obesidade. Apesar da alta mortalidade dessa doença, as doenças do fígado não afetam todas as subpopulações de forma igual. A detecção precoce da patologia é determinante para o desfecho dos pacientes, mas as pacientes do sexo feminino parecem ser marginalizadas quando se trata do diagnóstico precoce de doenças hepáticas.

O conjunto de dados é composto por 584 registros de pacientes coletados no nordeste de Andhra Pradesh, na Índia. A tarefa de predição consiste em determinar se um paciente sofre de doença hepática com base em informações sobre diversos marcadores bioquímicos, incluindo albumina e outras enzimas necessárias para o metabolismo.

### 🧠 Modelo Final: Random Forest com SMOTE

**Link do Notebook**: [Google Colab](https://colab.research.google.com/drive/1hcM9gq6GKSIyd4yXzhrtj6E1O54fqEPE?usp=sharing) 📝

Considerando a finalidade acadêmica, e sabedores da sensibilidade em modelos voltados para área de saúde, tentamos duas últimas ações:
    - Aplicar o `SMOTE` com objetivo de mitigar o desbalanceamento das classes.
    - Utilizar o `StratifiedKFold`, que performa melhor com dados binários.

### 📈 Conclusões Fase 1
Os resultados aplicando o `SMOTE` mostraram-se muito mais assertivos nas validações do modelo. O **"RandomForest SMOTE"** apresentou:
    - Capacidade de generalização
    - Melhor detecção da classe positiva
    - Maior acurácia no teste real

O modelo agora é excelente em identificar **'Não pacientes'**. Dos 33 reais, ele acertou 30 (Verdadeiros Negativos) e errou apenas 3 (Falsos Positivos).

No entanto, o custo dessa melhoria foi que o modelo agora erra mais na classe **'Paciente'**. Dos 83 reais, ele acertou 60 (Verdadeiros Positivos) e errou 23 (Falsos Negativos).

Em caso de evolução futura deste modelo, uma nova estratégia seria revalidar os hiperparâmetros.

| Modelo           | Acurácia | F1-Score | CV Score |
|------------------|----------|----------|----------|
|RandomForest SMOTE| 0.801724 | 0.843537 | 0.734239 |

Com base na tabela dos resultados finais, o modelo `RandomForest` com `SMOTE` apresenta o melhor desempenho geral, alcançando a maior acurácia e o maior score de validação cruzada.
