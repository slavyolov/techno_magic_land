from src.data_preparation import DataPreparation
from pyhocon import ConfigFactory
from linear_model import LinearModel


config = ConfigFactory.parse_file('config/COMMON.conf')


if __name__ == '__main__':
    data_preparation = DataPreparation(config=config)
    tables, visitors_ts, school_holidays, combined_df = data_preparation.run()

    # data profiling (run once)
    # data_preparation.data_profiling(tables)

    # Train model :
    lr = LinearModel(config=config, input_data=combined_df)
    lr.train(version=1)
    lr.predict()

    lr.train(version=2)
    lr.predict()

    lr.train(version=3)
    lr.predict()
    # Predictions


    # Other experiments :
    # https://colab.research.google.com/drive/1rAu9GthiODGt1sWbXPwJj9T3bJ7NuUGI#scrollTo=xn2veWYXSoOh&uniqifier=1