import seaborn as sns
import statsmodels.api as sm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error as mae


class LinearModel:
    def __init__(self, input_data, config):
        self.config = config
        self.input_data = input_data
        self.target_variable = self.config.lr_model.target
        self.transformed_target = self.config.lr_model.transformed_target
        self.features = self.config.lr_model.features
        self.train_set = None
        self.test_set = None
        self.model = None

    def prepare_input_data(self, get_plots=True):
        if get_plots:
            self._dist_plot(variable_name=self.target_variable)
            self._dist_plot(variable_name="wind_speed_kph_avg")
            self._dist_plot(variable_name="temperature_celsius_avg")
            self._dist_plot(variable_name="pressure_in_avg")
            self._dist_plot(variable_name="visitors_count_lag_7")
            self._dist_plot(variable_name="visitors_count_lag_1")
            self._dist_plot(variable_name="visitors_count_mean")
            self._dist_plot(variable_name="visitors_count_min")
            self._dist_plot(variable_name="visitors_count_max")
            self._dist_plot(variable_name="visitors_count_std")
            self._dist_plot(variable_name="visitors_count_median")

        # Log Transform the target variable because it is right skewed
        self.input_data[self.transformed_target] = np.log(self.input_data[self.target_variable])
        self.input_data["log_visitors_count_lag_1"] = np.log(self.input_data["visitors_count_lag_1"])
        self.input_data["log_visitors_count_lag_2"] = np.log(self.input_data["visitors_count_lag_2"])
        self.input_data["log_visitors_count_lag_3"] = np.log(self.input_data["visitors_count_lag_3"])
        self.input_data["log_visitors_count_lag_7"] = np.log(self.input_data["visitors_count_lag_7"])
        self.input_data["log_visitors_count_mean"] = np.log(self.input_data["visitors_count_mean"])
        self.input_data["log_visitors_count_min"] = np.log(self.input_data["visitors_count_min"])
        self.input_data["log_visitors_count_max"] = np.log(self.input_data["visitors_count_max"])
        self.input_data["log_visitors_count_std"] = np.log(self.input_data["visitors_count_std"])
        self.input_data["log_visitors_count_median"] = np.log(self.input_data["visitors_count_median"])

        self.input_data["log_visitors_count_lag_5"] = np.log(self.input_data["visitors_count_lag_5"])
        self.input_data["log_visitors_count_lag_9"] = np.log(self.input_data["visitors_count_lag_9"])
        self.input_data["log_visitors_count_lag_10"] = np.log(self.input_data["visitors_count_lag_10"])
        self.input_data["log_visitors_count_lag_11"] = np.log(self.input_data["visitors_count_lag_11"])

        if get_plots:
            self._dist_plot(variable_name=self.transformed_target)

        # split the data by keeping the temporal order
        train_test_split = int(self.config.lr_model.train_fraction * len(self.input_data))

        # drop na rows
        self.input_data.dropna(how="any", inplace=True)

        self.train_set = self.input_data[:train_test_split]
        self.test_set = self.input_data[train_test_split:]

        print("input data was prepared!")

    def train(self):
        # prepare input data :
        self.prepare_input_data(get_plots=False)

        # Statistics for the training :
        print("Target variable model with the following features : ", self.transformed_target)
        print("Features model with the following features : ", self.features)
        print("Training set size : ", len(self.train_set))

        # Train the model
        y = self.train_set[[self.transformed_target]]

        # 1st iteration - test with all prepared features
        X = sm.add_constant(self.train_set[self.features])
        model = sm.OLS(endog=y, exog=X).fit()

        # Store the coefficients
        with open('src/output/model/summary_all_features.txt', 'w') as fh:
            fh.write(model.summary().as_text())

        # 2nd iteration - select optimal features after backward elimination
        best_features = [
            'log_visitors_count_mean',
            'log_visitors_count_lag_1',
            'log_visitors_count_std',
            'log_visitors_count_min',
            'log_visitors_count_max',
        ]

        X_best = sm.add_constant(self.train_set[best_features])
        self.model = sm.OLS(endog=y, exog=X_best).fit()

        # Store the coefficients
        with open('src/output/model/summary_best_features.txt', 'w') as fh:
            fh.write(self.model.summary().as_text())

        return model

    def predict(self):
        best_features = [
            'log_visitors_count_mean',
            'log_visitors_count_lag_1',
            'log_visitors_count_std',
            'log_visitors_count_min',
            'log_visitors_count_max',
        ]

        # Train the model
        y_test = self.test_set[[self.transformed_target]]

        # 1st iteration - test with all prepared features
        X_test = sm.add_constant(self.test_set[best_features])

        # predictions
        y_pred = self.model.predict(X_test)

        evaluate = pd.DataFrame(y_test)
        evaluate = evaluate.rename(columns={'log_visitors_count': 'y_test'})
        evaluate["y_pred"] = y_pred

        # error = mae(y_test, y_pred) # TODO: check

        #TODO: backtransformation of the loagrithm - todo check if true
        evaluate["y_test"] = np.exp(evaluate["y_test"])
        evaluate["y_pred"] = np.exp(evaluate["y_pred"])

        # print(error)
        print(evaluate.plot())

    def _dist_plot(self, variable_name: str):
        fig = plt.figure(figsize=(10, 4))
        sns.distplot(self.input_data[variable_name])
        fig.savefig(f'src/output/plots/modeling/{variable_name}_dist_plot.png', bbox_inches="tight")
        plt.close()

    # https://colab.research.google.com/drive/1rAu9GthiODGt1sWbXPwJj9T3bJ7NuUGI#scrollTo=xn2veWYXSoOh&uniqifier=1

