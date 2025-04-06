from kedro.pipeline import Pipeline, node, pipeline
from .nodes import preparar_dados

def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        node(
            func=preparar_dados,
            inputs="dataset_kobe_dev",
            outputs="data_filtered",
            name="preparar_dados_node"
        )
    ])
