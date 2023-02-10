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

def grid_search_arima(y):
    p = d = q = range(0, 7)
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



grid_search_arima(y)

# The set of parameters with the minimum AIC is: SARIMA(2, 1, 6) - AIC:3336.1013657258054

import numpy as np
sar = sm.tsa.statespace.SARIMAX(y,
                                order=(1,1,1)).fit(max_iter = 50, method = 'powell')
print(sar.summary())


# Data
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

# Predictions
y_pred = sar.forecast(steps = 30)

evaluate = pd.DataFrame(test_set[:30])
evaluate["y_pred"] = y_pred
evaluate = evaluate[["visitors_count", "y_pred"]]
print(evaluate.plot())
