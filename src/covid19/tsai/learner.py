import os

from fastai.callback.all import SaveModelCallback, CSVLogger, EarlyStoppingCallback, ReduceLROnPlateau
from fastai.learner import Learner
from fastai.losses import MSELossFlat, L1LossFlat
from torch.nn import MSELoss, L1Loss, NLLLoss
from fastai.metrics import mae, mse

from covid19.config import CHECKPOINTS_DIR
from covid19.metrics import mape, smape, rmse, mape2
import numpy as np

from fastai.distributed import *


class TSAILearner(Learner):
    def __init__(self, dls, model, early_stop_patience: int, work_dir: str, predict: bool):
        super().__init__(
            dls=dls,
            model=model,
            cbs=[
                CSVLogger(),
                SaveModelCallback(with_opt=True),
                ReduceLROnPlateau(patience=5, min_lr=1e-4),
                EarlyStoppingCallback(min_delta=0, patience=early_stop_patience)
            ],
            metrics=[
                mse, mae, rmse,
                smape,
                mape
            ],
            path=work_dir,
            model_dir='',
            loss_func=L1LossFlat()
        )
