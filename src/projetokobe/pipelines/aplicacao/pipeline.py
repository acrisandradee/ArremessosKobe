from kedro.pipeline import Pipeline, node, pipeline
from .nodes import aplicar_modelo_producao

def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=aplicar_modelo_producao,
                inputs="dataset_kobe_prod_path",  # <- nome no catalog.yml
                outputs="predicoes_prod_path",
                name="aplicacao_modelo_node",
            )
        ]
    )
