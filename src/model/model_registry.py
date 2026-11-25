 
from model.model import (
    cellsplicenet,
    deltacellsplicenet,
)
# Additive: vanilla ViT baselines (ablation models). See model/vit_baseline.py.
from model.vit_baseline import (
    vitcsn_seq,
    vitcsn_all,
)
# Additive: CellSpliceNet SeqOnly baseline. See model/seqonly_baseline.py.
from model.seqonly_baseline import (
    csn_seqonly,
    csn_full_nocues,
    csn_full_nocues_strict,
    csn_full_nocues_nostruct,
    csn_nostruct,
    csn_noexpr,
    csn_full_nocues_noexpr,
    csn_seqonly_nocues,
)

MODEL_LIST = [
    cellsplicenet,
    deltacellsplicenet,
    vitcsn_seq,
    vitcsn_all,
    csn_seqonly,
    csn_full_nocues,
    csn_full_nocues_strict,
    csn_full_nocues_nostruct,
    csn_nostruct,
    csn_noexpr,
    csn_full_nocues_noexpr,
    csn_seqonly_nocues,
    ]


MODEL_DICT = {x.__name__.lower(): x for x in MODEL_LIST}


def str2model(model_name):

    try:
        model = MODEL_DICT[model_name]

    except:
        raise NotImplementedError(f"{model_name} not implemented. Check model registry")

    return model
