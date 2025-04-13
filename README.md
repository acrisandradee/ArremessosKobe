# 🏀 Projeto de Predição de Arremessos - Kobe Bryant

## Este projeto aplica técnicas de Machine Learning para prever se um arremesso feito por Kobe Bryant durante sua carreira resultou em cesta ou não.

## 🛠️ Tecnologias Utilizadas
- 🐍 Python
- 📊 Scikit-learn
- 🧪 PyCaret
- 🚀 MLFlow
- 📈 Streamlit
  <p align="center">
  <img src="https://img.shields.io/badge/Kaggle-20BEFF?style=for-the-badge&logo=Kaggle&logoColor=white" />
  <img src="https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=Streamlit&logoColor=white" />
  <img src="https://img.shields.io/badge/scikit_learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white" />
  <img src="https://img.shields.io/badge/Python-FFD43B?style=for-the-badge&logo=python&logoColor=blue" />
  <img src="https://img.shields.io/badge/VSCode-0078D4?style=for-the-badge&logo=visual%20studio%20code&logoColor=white" />
  <img src="https://img.shields.io/badge/Colab-F9AB00?style=for-the-badge&logo=googlecolab&color=525252" />
</p>

## 🔁 Como Rodar o Projeto
  1. Clone o Repositório

         git clone https://github.com/acrisandradee/ArremessosKobe.git

      
  2. Ative o ambiente virtual
     
             python -m venv .venv
             source .venv/bin/activate  # Linux/macOS
             .venv\Scripts\activate   # Windows

  3. Instale as dependências do projeto

          pip install -r requirements.txt
 
      
  4. Execute as pipelines com Kedro
     
    kedro run --pipeline=PreparacaoDados
    kedro run --pipeline=treinamento
    kedro run --pipeline=aplicacao

5. Execute o MLFlow
   
        Mlflow ui

6. Execute o Dashboard no Streamlit
   
        streamlit run dashboard/app.py
   
![image](https://github.com/user-attachments/assets/5994f80b-d5d8-4a4b-9830-931c688c1ae4)


# 💡 Como as Ferramentas Auxiliam no Pipeline

## Streamlit
  - O Streamlit criou um dashboard interativo para simulacao de arremessos, permitindo o monitoramento do modelo com visualizacoes graficas.
![image](https://github.com/user-attachments/assets/33fa7c91-4780-445f-bc17-c412c4b09db2)

 ## MLFlow
 - O Mlflow foi essencial para acompanhar todas etapas do ciclo de vida do modelo. Durante os experimentos ele registrou as metricas,parametros utilizados e os modelos gerados.
 - O modelo final foi registrado como ModeloArremessoKobe@prod, permitindo que ele fosse facilmente acessado e reutilizado. Quando necessário, o modelo pode ser atualizado com novas versões, mantendo o histórico completo no Model Registry.
 -  Além disso, o MLflow também foi usado para monitorar o desempenho do modelo em producoes, com o registro de metricas como F1 Score e Log Loss, garantindo visibilidade sobre a saúde do modelo em tempo real.
![image](https://github.com/user-attachments/assets/7b0e8487-b4f0-445d-8780-92a3129a876f)


## Pycaret
-O PyCaret facilitou o treinamento dos modelos ao automatizar tarefas como normalização dos dados, balanceamento das classes e seleção das variáveis mais relevantes.
-  Além disso, ele permitiu testar e comparar diferentes algoritmos de forma rápida, com avaliação baseada em metricas como F1 Score e Log Loss, facilitando a escolha do modelo com melhor desempenho.

![image](https://github.com/user-attachments/assets/22d9ca26-36f1-483f-aed8-158bb507173f)

## Scikit-Learn
- O Scikit-Learn serviu como base para algoritimos usados como a regressao logistica e a arvore de decisao. mesmo usado com o PyCaret, ele garantiu estabilidade, desempenho e compatibilidade com o MLflow, permitindo que os modelos fossem treinados, avaliados e salvos.

 ![image](https://github.com/user-attachments/assets/8ddbeb41-0bc1-4de0-a834-db7c63487b25)
 
# 📁 Artefatos que foram gerados 
| Etapa do Pipeline           | Artefato                          | Caminho                          | Descrição detalhada |
|----------------------------|-----------------------------------|----------------------------------|----------------------|
| Coleta (dados brutos)      | dataset_kobe_dev.parquet          | data/01_raw/                     | Dados históricos de arremessos realizados por Kobe Bryant com múltiplas variáveis.   |
| Coleta (dados de produção) | dataset_kobe_prod.parquet         | data/01_raw/                     | Dados de arremessos simulados para aplicação em produção, com mesma estrutura da base de desenvolvimento. |
| Preparação dos dados       | data_filtered.parquet             | data/02_intermediate/            | Dados filtrados contendo apenas as colunas relevantes e sem valores nulos. Base pronta para modelagem. |
| Separação treino/teste     | base_train.parquet                | data/processed/                  | Subconjunto com 80% dos dados usados para treinar os modelos, com estratificação da variável alvo. |
| Separação treino/teste     | base_test.parquet                 | data/processed/                  | Subconjunto com 20% dos dados usados para avaliar o desempenho dos modelos treinados. |
| Treinamento do modelo      | modelo_vencedor/ (pasta com pkl)  | data/06_models/                  | Modelo final salvo com estrutura MLflow (inclui model.pkl, MLmodel, conda.yaml, requirements.txt). |
| Aplicação em produção      | predicoes_prod.parquet            | data/07_model_output/            | Resultado da predição sobre a base de produção com as colunas `score` e `shot_made_flag_predito`. |
| Visualização / Dashboard   | previsoes_producao.parquet        | data/08_reporting/               | Dados de predições formatados para uso no dashboard de monitoramento com Streamlit. |

![image](https://github.com/user-attachments/assets/0aa0cbc9-e510-4965-9607-7f17a28f0596)


## 🔍Observacao sobre o modelo na nova base
- Observei que ha diferencas entre a base de treino principalmente arremessos curtos e a base de produção erremessos de 3 pontos. Essa diferença na distribuição foi evidenciada via histogramas comparativos, e impactou diretamente a performance em produção.
- Exemplo: F1 Score caiu de 0.72 para próximo de zero em testes anteriores.

  ![image](https://github.com/user-attachments/assets/e65b0a21-b11d-41a4-9f9a-94132bb7ccbc)

# 📊 Monitoramento da Saúde do Modelo

## Com a variavel resposta
Quando temos a informação se o arremesso foi certo ou errado (shot_made_flag), podemos:

- Cálcular  métricas como F1 Score e Log Loss
- Comparar valores reais e preditos, permitindo detectar erros e padrões de desempenho
- Identificar mudanças no comportamento dos dados com o tempo 

## Sem a variavel resposta
Quando não sabemos se o arremesso foi acertado:

- Analisamos a distribuição dos scores (confiança nas previsões) para ver se o modelo continua coerente
- Usamos dashboards e logs para acompanhar como o modelo está se comportando

# 🔁 Estratégias de Retreinamento do Modelo
## Estrategia Reativa
Essa abordagem e usada quando observamos uma queda significativa nas métricas de desempenho, como F1 Score ou Log Loss.
- tipo se o modelo comecar a erra muito nos arremessos
- Entao coletamos novos dados rotulados, atualizamos a base de treino e reexecutamos o pipeline para gerar uma nova versão do modelo.

  ## Estrategia Preditiva
  Neste o retreinamento é feito de forma periódica ou automática, mesmo que o modelo ainda esteja performando bem.
  - Atualizamos o modelo a cada epoca com dados mais recentes
  -  ou pipeline pode ser agendado para incluir novos dados. reavaliar a performance e se estiver melhor, publicar uma nova versão no MLflow Registry.

  
 
---

<div align="center">

✨ Desenvolvido por **Cristina Andrade** – 2025  
   Projeto de **Engenharia de Machine Learning** baseado no Framework TDSP (Microsoft)  
   Da ideia ao deploy com **Kedro**, **PyCaret**, **MLflow** e **Streamlit**

</div>

  

