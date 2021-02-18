import os

from fastai.callback.all import SaveModelCallback, CSVLogger, EarlyStoppingCallback
from fastai.learner import Learner
from fastai.metrics import mae, mse

from covid19.config import CHECKPOINTS_DIR
from covid19.metrics import mape, smape, rmse, mape2

from fastai.distributed import *


class TSAILearner(Learner):
    def __init__(self, dls, model, early_stop_patience: int, work_dir: str, predict: bool):
        super().__init__(
            dls=dls,
            model=model,
            cbs=[
                CSVLogger(),
                SaveModelCallback(with_opt=True),
                EarlyStoppingCallback(min_delta=0, patience=early_stop_patience)
            ],
            metrics=[mse, mae, rmse, smape, mape],
            path=work_dir,
            model_dir=''
        )
