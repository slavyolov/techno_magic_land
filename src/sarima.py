import pandas as pd


df = pd.read_excel("src/output/data/tml_datamart.xlsx")
print(df.shape)
df.head()

# Split the data :

df_ts = df[["Дата", "visitors_count"]]

# split the data by keeping the temporal order
train_test_split = int(0.8 * len(df_ts))

# drop na rows
df_ts.dropna(how="any", inplace=True)

train_set = df_ts[:train_test_split]
test_set = df_ts[train_test_split:]

y = train_set['visitors_count']

import statsmodels.api as sm
import itertools
import warnings

warnings.filterwarnings("ignore")

# fit SARIMA based on helper plots
import numpy as np
import statsmodels.api as sm
import itertools
import warnings
warnings.filterwarnings("ignore")


#
# The set of parameters with the minimum AIC is: SARIMA(2, 1, 2)x(0, 2, 2, 12) - AIC:2851.2651401210264
sar = sm.tsa.statespace.SARIMAX(np.log(y),
                                order=(2,1,2),
                                seasonal_order=(0,2,2,12)).fit(max_iter = 50, method = 'powell')
print(sar.summary())


# Season 12  : The set of parameters with the minimum AIC is: SARIMA(0, 1, 2)x(2, 2, 2, 7) - AIC:3134.815782921719

# evaluate an ARIMA model for a given order (p,d,q) and return RMSE
# grid search ARIMA parameters for time series
import warnings
from pandas import Series
from statsmodels.tsa.arima_model import ARIMA
from sklearn.metrics import mean_squared_error
from math import sqrt
import numpy



def sarima_grid_search(y, seasonal_period):
    p = d = q = range(0, 3)
    pdq = list(itertools.product(p, d, q))
    seasonal_pdq = [(x[0], x[1], x[2], seasonal_period) for x in list(itertools.product(p, d, q))]

    mini = float('+inf')

    for param in pdq:
        for param_seasonal in seasonal_pdq:
            try:
                mod = sm.tsa.statespace.SARIMAX(y,
                                                order=param,
                                                seasonal_order=param_seasonal,
                                                enforce_stationarity=False,
                                                enforce_invertibility=False)

                results = mod.fit()

                if results.aic < mini:
                    mini = results.aic
                    param_mini = param
                    param_seasonal_mini = param_seasonal

                print('SARIMA{}x{} - AIC:{}'.format(param, param_seasonal, results.aic))
            except:
                continue
    print('The set of parameters with the minimum AIC is: SARIMA{}x{} - AIC:{}'.format(param_mini, param_seasonal_mini,
                                                                                       mini))


sarima_grid_search(y=y, seasonal_period=3)

# SARIMA(2, 2, 2)x(2, 2, 2, 7) - AIC:3159.945498282935

# The set of parameters with the minimum AIC is: SARIMA(2, 1, 2)x(1, 1, 2, 6) - AIC:3159.5083571141586
# The set of parameters with the minimum AIC is: SARIMA(0, 1, 2)x(2, 2, 2, 7) - AIC:3134.815782921719


# Predictions
y_pred = sar.forecast(steps = 30)
y_pred = np.exp(y_pred)
evaluate = pd.DataFrame(test_set[:30])
evaluate["y_pred"] = y_pred
evaluate = evaluate[["visitors_count", "y_pred"]]
print(evaluate.plot())



# BOXPLOT :
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


df = pd.read_excel("src/output/data/tml_datamart.xlsx")
print(df.shape)

df_2018 = df[df["Дата"].between('2018-01-01', '2018-12-31')]
df_2019 = df[df["Дата"].between('2019-01-01', '2019-12-31')]
df_2020 = df[df["Дата"].between('2020-01-01', '2020-12-31')]


print(len(df_2019))
fig, ax = plt.subplots()
fig.set_size_inches((12,4))
sns.boxplot(x='month', y='visitors_count', data=df_2018, ax=ax)
plt.show()

fig, ax = plt.subplots()
fig.set_size_inches((12,4))
sns.boxplot(x='month', y='visitors_count', data=df_2019, ax=ax)
plt.show()

fig, ax = plt.subplots()
fig.set_size_inches((12,4))
sns.boxplot(x='month', y='visitors_count', data=df_2020, ax=ax)
plt.show()

def boxplot_per_period(data, year):
    data = data[data["Дата"].between(f'{year}-01-01', f'{year}-12-31')].copy(deep=True)

    fig, ax = plt.subplots()
    fig.set_size_inches((12, 4))
    sns.boxplot(x='month', y='visitors_count', data=data, ax=ax)
    fig.savefig(f'src/output/plots/eda/box_plot_per_month_for_{year}.png', bbox_inches="tight")
    plt.close()

boxplot_per_period(data=df, year=2018)
boxplot_per_period(data=df, year=2019)
boxplot_per_period(data=df, year=2020)

# Outlier detection :

# from pyod.models.ecod import ECOD
# clf = ECOD()
#
# y_arr = train_set['visitors_count'].values.reshape(-1,1)
# clf.fit(y_arr)

df = pd.read_excel("src/output/data/tml_datamart.xlsx")

# Get the median

df["year"] = df["Дата"].dt.year
df_median_per_month = df.groupby(["year", "month"])["visitors_count"].median().reset_index()
df_median_per_month = df_median_per_month.rename(columns={"visitors_count": "visitors_month_median"})
df = pd.merge(df, df_median_per_month, on=["year", "month"], how="inner")

# y - median
df["absolute_diff_from_median"] = abs(df["visitors_count"]-df["visitors_month_median"])
df_mad_per_month = df.groupby(["year", "month"])["absolute_diff_from_median"].median().reset_index()
df_mad_per_month = df_mad_per_month.rename(columns={"absolute_diff_from_median": "median_absolute_deviation"})
df = pd.merge(df, df_mad_per_month, on=["year", "month"], how="inner")

# find the modified z_score
# Modified z-score = 0.6745(xi – x̃) / MAD
# where xi  == 'visitors_count'
#       x̃   == median
#       MAD == median absolute deviation
df["modified_z_score"] = 0.6745 * (df["visitors_count"] - df["visitors_month_median"]) / df['median_absolute_deviation']
df_filtered = df[df["modified_z_score"].between(-3.5, 3.5)]


print(df_mad_per_month)



# ARIMA :

# grid search ARIMA parameters for time series
import warnings
from math import sqrt
from pandas import read_csv
from pandas import datetime
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error


# create a differenced series
def difference(dataset, interval=1):
    diff = list()
    for i in range(interval, len(dataset)):
        value = dataset[i] - dataset[i - interval]
        diff.append(value)
    return numpy.array(diff)

# invert differenced value
def inverse_difference(history, yhat, interval=1):
    return yhat + history[-interval]

def evaluate_arima_model(X, arima_order):
    # prepare training dataset
    X = X.astype('float32')
    train_size = int(len(X) * 0.50)
    train, test = X[0:train_size], X[train_size:]
    history = [x for x in train]
    # make predictions
    predictions = list()
    for t in range(len(test)):
        # difference data
        months_in_year = 12
        diff = difference(history, months_in_year)
        model = ARIMA(diff, order=arima_order)
        model_fit = model.fit(trend='nc', disp=0)
        yhat = model_fit.forecast()[0]
        yhat = inverse_difference(history, yhat, months_in_year)
        predictions.append(yhat)
        history.append(test[t])
    # calculate out of sample error
    rmse = sqrt(mean_squared_error(test, predictions))
    return rmse


# evaluate combinations of p, d and q values for an ARIMA model
def evaluate_models(dataset, p_values, d_values, q_values):
    dataset = dataset.astype('float32')
    best_score, best_cfg = float("inf"), None
    for p in p_values:
        for d in d_values:
            for q in q_values:
                order = (p, d, q)
                try:
                    rmse = evaluate_arima_model(dataset, order)
                    if rmse < best_score:
                        best_score, best_cfg = rmse, order
                    print('ARIMA%s RMSE=%.3f' % (order, rmse))
                except:
                    continue
    print('Best ARIMA%s RMSE=%.3f' % (best_cfg, best_score))


# evaluate parameters
p_values = range(0, 10)
d_values = range(0, 3)
q_values = range(0, 3)
warnings.filterwarnings("ignore")

import pandas as pd
df = pd.read_excel("src/output/data/tml_datamart.xlsx")
y = df["visitors_count"].values
evaluate_models(df["visitors_count"], p_values, d_values, q_values)







import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from skforecast.model_selection_statsmodels import grid_search_sarimax
import numpy as np
import statsmodels.api as sm
import itertools
import warnings
warnings.filterwarnings("ignore")


p = d = q = range(0, 3)
pdq = list(itertools.product(p, d, q))
param_grid = {'order': [(12, 0, 0), (12, 2, 0), (12, 1, 0), (12, 1, 1), (14, 1, 4)],
             'seasonal_order': [(0, 0, 0, 0)],
             'trend': [None, 'n', 'c']}


df = pd.read_excel("src/output/data/tml_datamart.xlsx")
data_my = df[['Дата', 'visitors_count']]
data_my = data_my.set_index('Дата')
data_my = data_my['visitors_count']
data_my = data_my.sort_index()

data_train = data_my[:380]
data_test = data_my[380:]

results_grid = grid_search_sarimax(
                y = data,
                param_grid = param_grid,
                initial_train_size = len(data),
                steps = 7,
                metric = 'mean_absolute_error',
                refit = False,
                verbose = False,
                fit_kwargs = {'maxiter': 200, 'disp': 0}
             )

print(results_grid.to_markdown(tablefmt="github", index=False))







def grid_search_arima(y):
    p = d = q = range(0, 3)
    pdq = list(itertools.product(p, d, q))

    mini = float('+inf')

    for param in pdq:
        try:
            mod = sm.tsa.statespace.SARIMAX(y,
                                            order=param,
                                            enforce_stationarity=False,
                                            enforce_invertibility=False)

            results = mod.fit()

            if results.aic < mini:
                mini = results.aic
                param_mini = param

            print('SARIMA{} - AIC:{}'.format(param, results.aic))
        except:
            continue

        print('The set of parameters with the minimum AIC is: SARIMA{} - AIC:{}'.format(param_mini, mini))