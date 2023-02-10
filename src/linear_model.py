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
        self.model_2 = None
        self.model_3 = None
        self.best_features = None


    def prepare_input_data(self, get_plots=True):
        if get_plots:
            self._dist_plot(variable_name=self.target_variable)
            self._dist_plot(variable_name="wind_speed_kph_max")
            self._dist_plot(variable_name="temperature_celsius_max")
            self._dist_plot(variable_name="pressure_in_max")
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
            self._dist_plot(variable_name="wind_speed_kph_max")
            self._dist_plot(variable_name="temperature_celsius_max")
            self._dist_plot(variable_name="pressure_in_max")
            self._dist_plot(variable_name="log_visitors_count_lag_7")
            self._dist_plot(variable_name="log_visitors_count_lag_1")
            self._dist_plot(variable_name="log_visitors_count_mean")
            self._dist_plot(variable_name="log_visitors_count_min")
            self._dist_plot(variable_name="log_visitors_count_max")
            self._dist_plot(variable_name="log_visitors_count_std")
            self._dist_plot(variable_name="log_visitors_count_median")

        # split the data by keeping the temporal order
        train_test_split = int(self.config.lr_model.train_fraction * len(self.input_data))

        # drop na rows
        self.input_data.dropna(how="any", inplace=True)

        # Exclude the first month
        self.input_data = self.input_data[self.input_data["Дата"] >= '2018-07-01']

        # Split the data
        self.train_set = self.input_data[:train_test_split]
        self.test_set = self.input_data[train_test_split:]

        self.train_set = self._outlier_removal(from_both=False)

        print("input data was prepared!")

        self._outlier_removal(from_both=True).to_excel("src/output/data/tml_datamart.xlsx")
        print("Storing output file for additional modeling")

    def train(self, version):
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
        self.model = sm.OLS(endog=y, exog=X).fit()

        # Store the coefficients
        with open('src/output/model/summary_all_features.txt', 'w') as fh:
            fh.write(self.model.summary().as_text())

        # 2nd iteration - select optimal features after backward elimination
        self.best_features = [
            'log_visitors_count_mean',
            'log_visitors_count_lag_1',
            'log_visitors_count_std',
            'log_visitors_count_min',
            'log_visitors_count_max',
        ]

        X_best = sm.add_constant(self.train_set[self.best_features])
        self.model_2 = sm.OLS(endog=y, exog=X_best).fit()

        # Store the coefficients
        with open('src/output/model/summary_best_features.txt', 'w') as fh:
            fh.write(self.model_2.summary().as_text())

        # 3rd iteration : select optimal features after backward elimination
        self.new_features = [
            'log_visitors_count_lag_1',
            "log_visitors_count_median",
            "is_weekend",
        ]

        X_third = sm.add_constant(self.train_set[self.new_features])
        self.model_3 = sm.OLS(endog=y, exog=X_third).fit()

        # Store the coefficients
        with open('src/output/model/summary_3rd_version.txt', 'w') as fh:
            fh.write(self.model_3.summary().as_text())

    def predict(self):
        model_d = {self.model: self.features,
                   self.model_2: self.best_features,
                   self.model_3: self.new_features,
                   }
        for index, (model, features) in enumerate(model_d.items()):
            y_test = self.test_set[[self.transformed_target]]

            # 1st iteration - test with all prepared features
            X_test = sm.add_constant(self.test_set[features])

            # predictions
            y_pred = model.predict(X_test)

            evaluate = pd.DataFrame(y_test)
            evaluate = evaluate.rename(columns={'log_visitors_count': 'y_test'})
            evaluate["y_pred"] = y_pred

            # Backtransform
            evaluate["y_test"] = np.exp(evaluate["y_test"])
            evaluate["y_pred"] = np.exp(evaluate["y_pred"])

            # Get MAE
            error = mae(y_test, y_pred)

            # Get MAPE
            MAPE = self.mape(actual=evaluate["y_test"], pred=evaluate["y_pred"])

            self._predictions_plot(df=evaluate, error=MAPE, version=index)

    def _dist_plot(self, variable_name: str):
        fig = plt.figure(figsize=(10, 4))
        sns.distplot(self.input_data[variable_name])
        fig.savefig(f'src/output/plots/modeling/{variable_name}_dist_plot.png', bbox_inches="tight")
        plt.close()

    def _outlier_removal(self, from_both: False):
        """
        Filter outliers using Modified Z-score filtering.

        Note that the process is calculated for every year-month. Therefore we have z-score that differs for every
        year-month

        https://hwbdocuments.env.nm.gov/Los%20Alamos%20National%20Labs/TA%2054/11587.pdf

        :param from_both: Remove the outliers from the Train set only or from both
        :return:
        """

        if from_both:
            df = self.input_data
        else:
            df = self.train_set

        # Get the median

        df["year"] = df["Дата"].dt.year
        df_median_per_month = df.groupby(["year", "month"])["visitors_count"].median().reset_index()
        df_median_per_month = df_median_per_month.rename(columns={"visitors_count": "visitors_month_median"})
        df = pd.merge(df, df_median_per_month, on=["year", "month"], how="inner")

        # y - median
        df["absolute_diff_from_median"] = abs(df["visitors_count"] - df["visitors_month_median"])
        df_mad_per_month = df.groupby(["year", "month"])["absolute_diff_from_median"].median().reset_index()
        df_mad_per_month = df_mad_per_month.rename(columns={"absolute_diff_from_median": "median_absolute_deviation"})
        df = pd.merge(df, df_mad_per_month, on=["year", "month"], how="inner")

        # find the modified z_score
        # Modified z-score = 0.6745(xi – x̃) / MAD
        # where xi  == 'visitors_count'
        #       x̃   == median
        #       MAD == median absolute deviation
        df["modified_z_score"] = 0.6745 * (df["visitors_count"] - df["visitors_month_median"]) / df[
            'median_absolute_deviation']
        df_filtered = df[df["modified_z_score"].between(-3.5, 3.5)]

        # Plot the new distribution after the outlier removal
        # Plot the final time series
        fig = plt.figure(figsize=(10, 4))
        sns.lineplot(data=df_filtered, x="Дата", y="visitors_count")
        fig.savefig('src/output/plots/eda/visitors_count_ts_plot_after_outliers.png', bbox_inches="tight")
        plt.close()

        return df_filtered

    def _predictions_plot(self, df, error, version):
        fig, ax = plt.subplots()
        fig.set_size_inches((12, 4))
        plt.title(f"MAPE : {round(error, 2)}")
        sns.lineplot(data=df[['y_test', 'y_pred']])
        fig.savefig(f'src/output/model/predicitons_w_version_{version}.png', bbox_inches="tight")
        plt.close()

    @staticmethod
    def mape(actual, pred):
        actual, pred = np.array(actual), np.array(pred)
        return np.mean(np.abs((actual - pred) / actual)) * 100