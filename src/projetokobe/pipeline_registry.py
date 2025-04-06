from projetokobe.pipelines import PreparacaoDados,Treinamento

def register_pipelines():
    return {
        "PreparacaoDados": PreparacaoDados.create_pipeline(),
        "treinamento": Treinamento.create_pipeline(),
        "__default__": PreparacaoDados.create_pipeline() + Treinamento.create_pipeline()
    }

