from src.data_preparation import DataPreparation
from pyhocon import ConfigFactory


config = ConfigFactory.parse_file('config/COMMON.conf')
relabel = False


if __name__ == '__main__':
    data_preparation = DataPreparation(config=config)
    tables = data_preparation.run()

    # data profiling (run once)
    # data_preparation.data_profiling(tables)



