from src.data_preparation import DataPreparation
from pyhocon import ConfigFactory
from linear_model import LinearModel


config = ConfigFactory.parse_file('config/COMMON.conf')


if __name__ == '__main__':
    data_preparation = DataPreparation(config=config)
    tables, visitors_ts, school_holidays, combined_df = data_preparation.run()

    combined_df.to_excel("src/output/data/tml_datamart.xlsx")

    # data profiling (run once)
    # data_preparation.data_profiling(tables)

    # Train model :
    lr = LinearModel(config=config, input_data=combined_df)
    lr.train()

    # Predictions
    lr.predict()
