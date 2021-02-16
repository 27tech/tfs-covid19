from .experiment import Experiment
from .experiments_set import ExperimentSet

from tsai.models.InceptionTimePlus import InceptionTimePlus17x17

if __name__ == "__main__":
    e = ExperimentSet(
        models=[InceptionTimePlus17x17],
        lr=[1e-3],
        early_stop_patience=100,
        epochs=1000,
        features=[
            ['existing_nx'],
            ['delta_existing_nx'],
            ['existing_nx', 'delta_existing_nx'],
            ['existing_std', 'delta_existing_std'],
            ['existing_norm', 'delta_existing_norm'],
            ['existing_rob', 'delta_existing_rob'],

            ['existing_nx_all', 'delta_existing_nx_all'],
            ['existing_std_all', 'delta_existing_std_all'],
            ['existing_norm_all', 'delta_existing_norm_all'],
            ['existing_rob_all', 'delta_existing_rob_all'],

        ],
        targets=[
            ['delta_existing_nx']
        ],
        window=[56],
        horizon=[14],
        batch_size=[256],
        country_filter=[['Ukraine']],
        region_filter=[None],
        runs=3,
    )
    e.run()

