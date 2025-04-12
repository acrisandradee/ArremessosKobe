"""
This is a boilerplate pipeline 'PreparacaoDados'
generated using Kedro 0.19.12
"""
import pandas as pd
import mlflow

def preparar_dados(df: pd.DataFrame) -> pd.DataFrame:
    colunas = [
        "lat", "lon", "minutes_remaining", "period",
        "playoffs", "shot_distance", "shot_made_flag"
    ]
    df = df[colunas].dropna()

    mlflow.log_param("colunas_utilizadas", len(colunas))
    mlflow.log_metric("linhas_preprocessadas", df.shape[0])
    mlflow.log_metric("colunas_preprocessadas", df.shape[1])

    return df
from sklearn.model_selection import train_test_split

def separar_treino_teste(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    X = df.drop(columns=["shot_made_flag"])
    y = df["shot_made_flag"]

    df_train, df_test = train_test_split(
        df,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    return df_train, df_test
