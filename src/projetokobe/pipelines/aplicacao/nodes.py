import pandas as pd
import mlflow
from mlflow.sklearn import load_model
from sklearn.metrics import f1_score, log_loss
from pathlib import Path

def aplicar_modelo_producao(df: pd.DataFrame) -> pd.DataFrame:
    with mlflow.start_run(run_name="PipelineAplicacao", nested=True):
        colunas = ["lat", "lon", "minutes_remaining", "period", "playoffs", "shot_distance"]
        df = df[colunas + (["shot_made_flag"] if "shot_made_flag" in df.columns else [])]
        df = df.dropna(subset=colunas)

        modelo = load_model("models:/ModeloArremessoKobe@prod")

        y_pred = modelo.predict(df[colunas])
        y_proba = modelo.predict_proba(df[colunas])[:, 1]

        df_resultado = df.copy()
        df_resultado["score"] = y_proba
        df_resultado["shot_made_flag_predito"] = y_pred

        # 🔹 Logar artefato manualmente
        output_path = "data/07_model_output/predicoes_prod.parquet"
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        df_resultado.to_parquet(output_path, index=False)
        mlflow.log_artifact(output_path)

        # 🔹 Logar métricas se variável alvo estiver presente
        if "shot_made_flag" in df.columns:
            y_true = df["shot_made_flag"].dropna()
            y_pred_filt = y_pred[df["shot_made_flag"].notna()]
            y_proba_filt = y_proba[df["shot_made_flag"].notna()]
            f1 = f1_score(y_true, y_pred_filt)
            loss = log_loss(y_true, y_proba_filt)
            mlflow.log_metric("f1_score_prod", f1)
            mlflow.log_metric("log_loss_prod", loss)
            print(f"[📊] F1: {f1:.4f} | LogLoss: {loss:.4f}")
        else:
            print("[⚠️] Variável alvo ausente - métricas não foram calculadas.")

        return df_resultado


