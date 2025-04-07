from kedro.pipeline import Pipeline, node
from .nodes import treinar_melhor_modelo

def create_pipeline(**kwargs):
    return Pipeline([
        node(
            func=treinar_melhor_modelo,
            inputs=["base_train", "base_test"],
            outputs=None,
            name="treinamento_modelo_vencedor"
        )
    ])
