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

        # Log Transform the target variable because it is right skewed
        self.input_data[self.transformed_target] = np.log(self.input_data[self.target_variable])

        if get_plots:
            self._dist_plot(variable_name=self.transformed_target)

        # split the data by keeping the temporal order
        train_test_split = int(self.config.lr_model.train_fraction * len(self.input_data))

        # first difference feature :
        # TODO: check if needed
        # self.input_data["visitors_count_first_diff"] = self.input_data["visitors_count"].diff()
        # self.input_data.dropna(how="any", inplace=True)

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
        X = sm.add_constant(self.train_set[self.features])
        model = sm.OLS(endog=y, exog=X).fit()

        # Store the coefficients
        with open('src/output/model/summary.txt', 'w') as fh:
            fh.write(model.summary().as_text())

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
