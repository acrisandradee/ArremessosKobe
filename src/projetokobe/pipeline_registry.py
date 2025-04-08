# src/projetokobe/pipeline_registry.py
from projetokobe.pipelines import PreparacaoDados, Treinamento
from projetokobe.pipelines.aplicacao import pipeline as Aplicacao

def register_pipelines():
    return {
        "PreparacaoDados": PreparacaoDados.create_pipeline(),
        "treinamento": Treinamento.create_pipeline(),
        "aplicacao": Aplicacao.create_pipeline(),
        "__default__": PreparacaoDados.create_pipeline() + Treinamento.create_pipeline() + Aplicacao.create_pipeline()
    }
