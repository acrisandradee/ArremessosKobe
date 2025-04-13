import mlflow
import mlflow.sklearn
import pandas as pd
from pycaret.classification import setup, create_model, tune_model, get_config
from sklearn.metrics import log_loss, f1_score
from imblearn.over_sampling import SMOTE
from pathlib import Path
import shutil  # ✅ Adicionado para remover pasta existente
from pycaret.classification import finalize_model, save_model

def treinar_melhor_modelo(df_train: pd.DataFrame, df_test: pd.DataFrame) -> None:
    """
    Treina modelos com PyCaret, escolhe o melhor com base em F1 e Log Loss,
    registra no MLflow e Model Registry, e salva o modelo localmente.
    """
    X_test = df_test.drop("shot_made_flag", axis=1)
    y_test = df_test["shot_made_flag"]

    setup(
        data=df_train,
        target="shot_made_flag",
        session_id=42,
        normalize=True,
        transformation=True,
        fix_imbalance=True,
        fix_imbalance_method=SMOTE(),
        remove_outliers=True,
        remove_multicollinearity=True,
        multicollinearity_threshold=0.95,
        fold=10,
        fold_shuffle=True,
        html=False,
        verbose=False
    )

    print("🔧 Treinando modelo: Regressão Logística")
    lr_model = tune_model(create_model("lr"), optimize="F1")

    print("🔧 Treinando modelo: Árvore de Decisão")
    dt_model = tune_model(create_model("dt"), optimize="F1")

    pipeline = get_config("pipeline")
    X_test_transformed = pipeline.transform(X_test)

    # Avaliação Regressão Logística
    y_pred_lr = lr_model.predict(X_test_transformed)
    y_proba_lr = lr_model.predict_proba(X_test_transformed)
    f1_lr = f1_score(y_test, y_pred_lr)
    loss_lr = log_loss(y_test, y_proba_lr)

    # Avaliação Árvore
    y_pred_dt = dt_model.predict(X_test_transformed)
    y_proba_dt = dt_model.predict_proba(X_test_transformed)
    f1_dt = f1_score(y_test, y_pred_dt)
    loss_dt = log_loss(y_test, y_proba_dt)

    print(f"📊 Logística - F1 Score: {f1_lr:.4f} | Log Loss: {loss_lr:.4f}")
    print(f"📊 Árvore    - F1 Score: {f1_dt:.4f} | Log Loss: {loss_dt:.4f}")

    # Log de métricas
    mlflow.log_metrics({
        "f1_score_logistica": f1_lr,
        "log_loss_logistica": loss_lr,
        "f1_score_arvore": f1_dt,
        "log_loss_arvore": loss_dt
    })

    # Escolha do melhor modelo (score ponderado)
    score_lr = (0.7 * f1_lr) - (0.3 * loss_lr)
    score_dt = (0.7 * f1_dt) - (0.3 * loss_dt)

    if score_lr > score_dt:
        modelo_vencedor = "Logistica"
        modelo_final = lr_model
    else:
        modelo_vencedor = "Arvore"
        modelo_final = dt_model

    mlflow.log_param("modelo_vencedor", modelo_vencedor)

        # Registro no MLflow Model Registry
    result = mlflow.sklearn.log_model(
        sk_model=modelo_final,
        artifact_path="modelo_vencedor",
        registered_model_name="ModeloArremessoKobe"
    )

    # Atribuir alias 'prod' à versão mais recente
    from mlflow import MlflowClient
    client = MlflowClient()

    # Obtém a versão da última adição
    latest_versions = client.get_latest_versions("ModeloArremessoKobe", stages=["None"])
    if latest_versions:
        version = latest_versions[0].version
        client.set_registered_model_alias("ModeloArremessoKobe", "prod", version)
        print(f"🔁 Alias 'prod' atribuído à versão {version}")
    else:
        print("⚠️ Nenhuma versão encontrada para atribuir alias.")

    # Caminho local para salvar o modelo
    local_model_path = Path("data/06_models/modelo_vencedor")

    # ✅ Remove pasta anterior se já existir
    if local_model_path.exists():
        shutil.rmtree(local_model_path)

    # Cria pasta novamente
    local_model_path.mkdir(parents=True, exist_ok=True)

    # Salva o modelo localmente
    mlflow.sklearn.save_model(sk_model=modelo_final, path=str(local_model_path))
    modelo_final = finalize_model(modelo_final)
    save_model(modelo_final, model_name="modelo_vencedor")  

    print(f"✅ Modelo escolhido: {modelo_vencedor}")
    print(f"📦 Modelo salvo localmente em: {local_model_path.resolve()}")
    print("✅ Registro completo no MLflow e no Model Registry.")
