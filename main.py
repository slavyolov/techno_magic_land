from src.data_preparation import DataPreparation
from pyhocon import ConfigFactory
from linear_model import LinearModel


config = ConfigFactory.parse_file('config/COMMON.conf')


if __name__ == '__main__':
    data_preparation = DataPreparation(config=config)
    tables, visitors_ts, school_holidays, combined_df = data_preparation.run()

    combined_df.to_excel("src/output/data/tml_datamart.xlsx")

    # Plotting and additional check

    # data profiling (run once)
    # data_preparation.data_profiling(tables)

    # Train model :
    lr = LinearModel(config=config, input_data=combined_df)
    # lr.prepare_input_data()
    lr.train()
    fraction = 0.8


    def linear_model():
        import statsmodels.api as sm

        X_constant = sm.add_constant(X)
        lin_reg = sm.OLS(y, X_constant).fit()
        lin_reg.summary()

# Predictions :
