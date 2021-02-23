from covid19.models.inception import InceptionTimePlus17
from .experiment import Experiment
from .experiments_set import ExperimentSet
import calendar
from covid19.models import Transformer
from tsai.models.InceptionTimePlus import InceptionTimePlus17x17
from tsai.models.TST import TST
from tsai.models.RNNPlus import GRUPlus, LSTMPlus
from covid19.datasets.open_world import OpenWorldDataset

class LSTMPlus1(LSTMPlus):
    def __init__(self, args, **kwargs):
        super(LSTMPlus1, self).__init__(*args, **kwargs, hidden_size=512)

if __name__ == "__main__":
    # d = OpenWorldDataset()
    # d.filter_country(['Ukraine'])
    # print(d._dataframe)
    e = ExperimentSet(
        models=[
            # LSTMPlus,
            # GRUPlus
            # TST,
            InceptionTimePlus17x17
        ],
        lr=[1e-1],
        early_stop_patience=100,
        epochs=100000,
        features=[
            # ['total_cases_per_million_std', 'new_cases_per_million_std'], #0.160758
            # ['total_cases_per_population_std'],
            # ['total_cases_std', 'new_cases_std'] # 0.462105
            # ['new_cases_smoothed_nx'], 0.877701
            # ['total_cases_per_million_std'] # 0.20904667675495148
            # ['total_cases_per_population', 'new_cases_per_population'] # 0.131196
            # ['total_cases_per_population', 'new_cases_per_population', 'non_sick_per_population'], #0.293098
            # [
            #     'total_cases_per_population',
            #     'new_cases_per_population',
            #     'non_sick_per_population',
            #     'people_vaccinated_per_population'
            # ],
            [
                'total_cases', 'new_cases', 'new_vaccinations',
                'people_fully_vaccinated'
                # 'new_cases_smoothed_nx',
                # 'total_cases_per_population_std',
                # 'new_cases_per_population_std',
                # 'non_sick_per_population_std',
                # 'people_vaccinated_per_population_std',
                # 'people_fully_vaccinated_per_population_std',
                # 'new_vaccinations_per_population_std'
            ] + list(calendar.day_name),
            # ['existing_nx'],
            # ['existing_std', 'confirmed_std', 'delta_existing_std'] + list(calendar.day_name),
            # ['existing_pop', 'confirmed_pop', 'none_sick_pop', 'delta_existing_pop'],
            # ['existing_pop', 'confirmed_pop', 'none_sick_pop', 'delta_existing_pop'] + list(calendar.day_name)
            # ['delta_existing_norm'],
            # ['delta_existing_nx'],
            # ['delta_existing_std'],˚
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
            # ['new_cases_origin']
            # ['new_cases_std'],
            # ['total_cases_per_population_std'],
            ['new_cases_origin']
            # ['new_cases_std']
            # ['existing_std'],
            # ['existing_nx'],
            # ['delta_existing_std'],
            # ['delta_confirmed_nx'],
            # ['delta_existing_nx']
        ],
        window=[56], #list(7 * h for h in range(1, 30)),
        horizon=[7], #list(7 * h for h in range(1, 20)),
        batch_size=[64],
        country_filter=[
            # ['United States', 'Italy', 'France', 'Germany', 'United Kingdom', 'Australia', 'Canada'],
            # ['Italy'],
            # ['France'],
            ['United States'],

        ],
        # country_filter=[['Ukraine']],
        # region_filter=[['Kyiv']],
        region_filter=[None],
        runs=1,
        do_predict=False

    )
    e.run()

