from .experiment import Experiment
from .experiments_set import ExperimentSet
import calendar

from tsai.models.InceptionTimePlus import InceptionTimePlus17x17

if __name__ == "__main__":
    e = ExperimentSet(
        models=[InceptionTimePlus17x17],
        lr=[1e-3],
        early_stop_patience=200,
        epochs=1000,
        features=[
            # ['existing_nx'],
            ['delta_existing_rob'],
            # ['delta_existing_norm'],
            # ['delta_existing_nx'],
            # ['delta_existing_std'],
            # ['existing_std_all', 'confirmed_std_all', 'delta_confirmed_std_all', 'delta_recovered_std_all'] + list(calendar.day_name),
            # ['existing_std', 'delta_existing_std'],
            # ['existing_std_all', 'delta_existing_std_all'],
            # ['existing_norm', 'delta_existing_norm'],
            # ['existing_norm_all', 'delta_existing_norm_all'],
            # ['delta_existing_nx'],
            # ['existing_nx', 'delta_existing_nx'],
            # ['existing_std', 'delta_existing_std'],
            # ['existing_norm', 'delta_existing_norm'],
            # ['existing_rob', 'delta_existing_rob'],
            #
            # ['existing_nx_all', 'delta_existing_nx_all'],
            # ['existing_std_all', 'delta_existing_std_all'],
            # ['existing_norm_all', 'delta_existing_norm_all'],
            # ['existing_rob_all', 'delta_existing_rob_all'],

        ],
        targets=[
            # ['existing_std'],
            # ['existing_nx'],
            ['existing_rob']
            # ['delta_existing_nx']
        ],
        window=[56], #list(7 * h for h in range(1, 30)),
        horizon=[7], #list(7 * h for h in range(1, 20)),
        batch_size=[256],
        country_filter=[['Ukraine']],
        # region_filter=[['Kyiv']],
        region_filter=[None],
        runs=5,
    )
    e.run()

