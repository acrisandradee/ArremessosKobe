"""
This is a boilerplate pipeline 'PreparacaoDados'
generated using Kedro 0.19.12
"""
import pandas as pd

def preparar_dados(df: pd.DataFrame) -> pd.DataFrame:
    colunas = [
        "lat", "lon", "minutes_remaining", "period",
        "playoffs", "shot_distance", "shot_made_flag"
    ]
    df = df[colunas].dropna()
    return df
