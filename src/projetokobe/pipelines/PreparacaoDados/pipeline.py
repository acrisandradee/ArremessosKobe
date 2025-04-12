from kedro.pipeline import Pipeline, node, pipeline
from .nodes import preparar_dados, separar_treino_teste

def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        node(
            func=preparar_dados,
            inputs="dataset_kobe_dev",
            outputs="data_filtered",
            name="preparar_dados_node"
        ),
        node(
            func=separar_treino_teste,
            inputs="data_filtered",
            outputs=["base_train", "base_test"],
            name="separar_treino_teste_node"
        )
    ])
