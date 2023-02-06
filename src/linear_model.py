import pandas as pd
import seaborn as sns
import statsmodels.api as sm
import matplotlib.pyplot as plt
import numpy as np


class LinearModel:
    def __init__(self, input_data, config):
        self.config = config
        self.input_data = input_data
        self.target_variable = self.config.lr_model.target
        self.transformed_target = self.config.lr_model.transformed_target
        self.features = self.config.lr_model.features
        self.train_set = None
        self.test_set = None

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

        #TODO : maybe log transform the wind speed per kph ?

        if get_plots:
            self._dist_plot(variable_name=self.transformed_target)

        # split the data by keeping the temporal order
        train_test_split = int(self.config.lr_model.train_fraction * len(self.input_data))

        # first difference feature :
        # TODO: check if needed
        # self.input_data["visitors_count_first_diff"] = self.input_data["visitors_count"].diff()
        # self.input_data.dropna(how="any", inplace=True)

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

        self.features =[
            'log_visitors_count_mean',
            'log_visitors_count_lag_1',
            'log_visitors_count_std',
            'log_visitors_count_min',
            'log_visitors_count_max',

            # "log_visitors_count_lag_5",
            # "log_visitors_count_lag_9",
            # "log_visitors_count_lag_10",
            # "log_visitors_count_lag_11",


            # 'hot', 'warm', 'cool', 'cold',
            # # non relevant columns
            # 'visitors_count_lag_2', 'visitors_count_lag_3', 'visitors_count_lag_7',
            # 'is_weekend',
            # 'day_of_week_sin', 'day_of_week_cos',
            # # 'month',
            # 'month_sin', 'month_cos',
            # # 'quarter',
            # # 'weekday_3', 'weekday_4', 'weekday_5', 'weekday_6', 'weekday_7',
            # # 'season',
            # 'autumn', 'spring', 'summer',
            # # 'is_public_holiday',
            # 'temperature_celsius_max',
            # 'wind_speed_kph_max',
            # 'pressure_in_max',
            # # 'hot', 'warm', 'cool', 'cold',
            # # 'is_summer_holiday',
            # # 'is_winter_holiday', 'is_spring_holiday', 'school_exams',
            # # 'is_school_holiday',
            # 'school_day',
            # 'log_visitors_count',
            # 'log_visitors_count_lag_1', 'log_visitors_count_lag_2',
            # 'log_visitors_count_lag_3', 'log_visitors_count_lag_7',
            # 'log_visitors_count_mean', 'log_visitors_count_min',
            # 'log_visitors_count_max', 'log_visitors_count_std',
            # 'log_visitors_count_median'
        ]

        X = sm.add_constant(self.train_set[self.features])
        model = sm.OLS(endog=y, exog=X).fit()

        # Store the coefficients
        with open('src/output/model/summary.txt', 'w') as fh:
            fh.write(model.summary().as_text())


        # Synergy :
        self.features =[
            'log_visitors_count_mean',
            'log_visitors_count_lag_1',
            'log_visitors_count_std',
            'log_visitors_count_min',
            'log_visitors_count_max',

            'is_weekend',
            'is_school_holiday',
            'is_public_holiday',
            'is_summer_holiday',
            'is_winter_holiday',
            'is_spring_holiday',
        ]
        X = sm.add_constant(self.train_set[self.features])
        from sklearn.preprocessing import PolynomialFeatures  # generating interaction terms
        x_interaction = PolynomialFeatures(2, interaction_only=True, include_bias=False).fit_transform(
            X)  # creating a new dataframe with the interaction terms included
        interaction_df = pd.DataFrame(x_interaction)

        X = sm.add_constant(interaction_df)
        sm.OLS(y, X).fit()

        # train random forest
        from sklearn.model_selection import GridSearchCV
        from sklearn.ensemble import RandomForestRegressor
        rf_regressor = RandomForestRegressor(n_estimators=100, random_state=0)

        # fit the regressor with x and y data
        features_ = [
            'visitors_day_of_week_mean',
            'visitors_count_lag_1',
            'is_weekend',
            'visitors_count_lag_2',
            'visitors_count_lag_3',
            'visitors_count_lag_7',
            "day_of_week",
            "is_school_holiday",
            'is_summer_holiday',
            'is_winter_holiday',
            'is_spring_holiday',
            'month'
        ]

        random_forest_tuning = RandomForestRegressor(random_state=42)
        param_grid = {
            'n_estimators': [100, 200, 500],
            'max_features': ['auto', 'sqrt', 'log2'],
            'max_depth': [4, 5, 6, 7, 8],
        }

        x = self.train_set[features_]
        y = self.train_set['visitors_count']

        GSCV = GridSearchCV(estimator=random_forest_tuning, param_grid=param_grid, cv=5)
        GSCV.fit(x, y)
        print(GSCV.best_params_)

        # Final model
        random_forest_tuned = RandomForestRegressor(**GSCV.best_params_, random_state=42).fit(x, y)
        y_pred = random_forest_tuned.predict(self.test_set[features_])
        y_test = self.test_set["visitors_count"]

        import pandas as pd
        evaluate = pd.DataFrame({"y_test": y_test, "y_pred": y_pred})

        from sklearn.metrics import mean_absolute_error as mae
        error = mae(y_test, y_pred)


        plt.barh(self.train_set[features_].columns, random_forest_tuned.feature_importances_)

        return model

    def predict(self):
        pass

    def validate(self):
        pass

    def generate_plots(self):
        pass

    def _dist_plot(self, variable_name: str):
        fig = plt.figure(figsize=(10, 4))
        sns.distplot(self.input_data[variable_name])
        fig.savefig(f'src/output/plots/modeling/{variable_name}_dist_plot.png', bbox_inches="tight")
        plt.close()

    # https://colab.research.google.com/drive/1rAu9GthiODGt1sWbXPwJj9T3bJ7NuUGI#scrollTo=xn2veWYXSoOh&uniqifier=1


    # TODO: tmp
    # sm.graphics.tsa.plot_acf(self.train_set['visitors_count'], lags=40)
    # sm.graphics.tsa.plot_pacf(self.train_set['visitors_count'], lags=40)
