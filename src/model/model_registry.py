 
from model.model import ( 
    cellsplicenet,
    deltacellsplicenet,
)

MODEL_LIST = [ 
    cellsplicenet, 
    deltacellsplicenet
    ]


MODEL_DICT = {x.__name__.lower(): x for x in MODEL_LIST}


def str2model(model_name):

    try:
        model = MODEL_DICT[model_name]

    except:
        raise NotImplementedError(f"{model_name} not implemented. Check model registry")

    return model
