# dashboard/app.py
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from mlflow.sklearn import load_model
import numpy as np



# Carrega predições anteriores (para painel)
import os
caminho = os.path.join("data", "07_model_output", "predicoes_prod.parquet")
df = pd.read_parquet(caminho)

st.set_page_config(page_title="Kobe - Monitoramento do Modelo", layout="wide")

st.title("🏀 Monitoramento de Arremessos do Kobe")
st.header("Simule uma nova jogada")
with st.form("form_predicao"):
    col1, col2 = st.columns(2)
    with col1:
        lat = st.number_input("Latitude", value=34.0443, format="%.4f")
        min_restante = st.slider("Minutos Restantes", 0, 12, value=10)
        playoffs = st.selectbox("É Playoff?", ["Não", "Sim"])
    with col2:
        lon = st.number_input("Longitude", value=-118.4268, format="%.4f")
        periodo = st.selectbox("Período do Jogo", list(range(1, 5)))
        distancia = st.slider("Distância do Arremesso", 0, 35, value=18)

    submit = st.form_submit_button("Prever Resultado")

if submit:
    st.subheader("Dados enviados ao modelo:")
    dados = pd.DataFrame.from_dict({
        "lat": [lat],
        "lon": [lon],
        "minutes_remaining": [min_restante],
        "period": [periodo],
        "playoffs": [1 if playoffs == "Sim" else 0],
        "shot_distance": [distancia]
    })
    st.write(dados)

    try:
        modelo = load_model("models:/ModeloArremessoKobe@prod")
        pred = modelo.predict(dados)[0]
        proba = modelo.predict_proba(dados)[0, 1]

        if pred == 1:
            st.success(f" Cesta convertida com {proba:.2%} de confiança!")
        else:
            st.error(f"❌ Errou a cesta. A bola bateu no aro e saiu. 😥")
        st.markdown("\n🧠 *Tente ajustar a posição ou a força do arremesso!*")

    except Exception as e:
        st.error(f"Erro ao carregar modelo: {e}")

# ==== Monitoramento geral ====
st.header(" Monitoramento da Operação")

# Métricas
if "shot_made_flag" in df.columns:
    st.subheader(" Desempenho")
    acertos = (df["shot_made_flag"] == df["shot_made_flag_predito"]).sum()
    total = len(df)
    f1 = round((2 * acertos) / (total + acertos), 4) if total > 0 else 0

    st.metric("F1 Score", f"{f1:.4f}")
    st.metric("Total predições", total)
else:
    st.warning("Dataset sem a coluna 'shot_made_flag'")

# Gráfico de dispersão
st.subheader("🗺️ Mapa dos Arremessos")
fig, ax = plt.subplots()
scatter = ax.scatter(df["lon"], df["lat"], c=df["score"], cmap="coolwarm", alpha=0.7)
ax.set_xlabel("Longitude")
ax.set_ylabel("Latitude")
ax.set_title("Dispersão dos Arremessos")
st.pyplot(fig)



# Tabela final
st.subheader(" Predições")
st.dataframe(df.sort_values("score", ascending=False).head(20))