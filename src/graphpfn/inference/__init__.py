from graphpfn.inference.finetune import predict_ft
from graphpfn.inference.icl import predict_icl
from graphpfn.inference.tune import tune_and_predict_ft
from graphpfn.inference.util import LossFn, ScoreFn

__all__ = [
    "LossFn",
    "ScoreFn",
    "predict_ft",
    "predict_icl",
    "tune_and_predict_ft",
]
