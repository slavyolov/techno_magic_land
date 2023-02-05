import pandas as pd
import seaborn as sns
from pandas_profiling import ProfileReport
import matplotlib.pyplot as plt
import numpy as np
from feature_engine.creation import CyclicalFeatures
from statsmodels.graphics.tsaplots import plot_pacf, plot_acf


class DataPreparation:
    def __init__(self, config):
        self.config = config

    def run(self):
        """
        Execute the data processing stages

        :return:
        """
        tables = self.read_data()

        visitors_ts = self.prepare_tml_visitors_count(tables=tables)
        school_holidays = self.prepare_school_holidays()
        weather_df = self.prepare_weather_data()
        # TODO: maybe filter from the TS holidays (Easter, Christmas, other - check the raw data ? )
        # TODO: or include flag. Check availability and working days of the centre among these days

        return tables, visitors_ts, school_holidays

    def read_data(self):
        xls = pd.ExcelFile('src/resources/TML_vistors_case_study_data.xlsx')

        df1 = pd.read_excel(xls, 'Таблица 1')
        df2 = pd.read_excel(xls, 'Таблица 2')
        df3 = pd.read_excel(xls, 'Таблица 3')

        return {
            'tml_visitors_p1': df1,
            'tml_visitors_p2': df2,
            'tml_visitors_p3': df3,
        }

    @staticmethod
    def data_profiling(tables: dict):
        # Profile reports :
        profile = ProfileReport(tables["tml_visitors_p1"], title="Profiling Report")
        profile.to_file(output_file=f'src/output/profiling/tml_visitors_p1.html')

        profile = ProfileReport(tables["tml_visitors_p2"], title="Profiling Report")
        profile.to_file(output_file=f'src/output/profiling/tml_visitors_p2.html')

        profile = ProfileReport(tables["tml_visitors_p3"], title="Profiling Report")
        profile.to_file(output_file=f'src/output/profiling/tml_visitors_p3.html')

        # profile = ProfileReport(tables["we_2018"], title="Profiling Report")
        # profile.to_file(output_file=f'src/output/profiling/we_2018.html')
        #
        # profile = ProfileReport(tables["we_2019"], title="Profiling Report")
        # profile.to_file(output_file=f'src/output/profiling/we_2019.html')
        #
        # profile = ProfileReport(tables["we_2020"], title="Profiling Report")
        # profile.to_file(output_file=f'src/output/profiling/we_2020.html')

        print("Profiling done!")

    def prepare_tml_visitors_count(self, tables):
        # Handle missing data for the 1st file
        tml_visitors_p1 = tables["tml_visitors_p1"].copy(deep=True)

        tml_visitors_p1["Точки за посещението"].fillna(0, inplace=True)
        tml_visitors_p1.loc[tml_visitors_p1['Дата на излизане'].isnull(),
                            'Дата на излизане'] = tml_visitors_p1['Дата на влизане']

        # drop the number column
        tml_visitors_p1 = tml_visitors_p1.drop('№', axis=1)

        # drop duplicates
        tml_visitors_p1 = tml_visitors_p1.drop_duplicates()

        # Aggregate on day level to get the intensity (Интензивност на посещения - брой посетители на дневна база)
        tml_visitors_count = tml_visitors_p1.groupby(by=["Дата"]).size().to_frame().reset_index()
        tml_visitors_count = tml_visitors_count.rename(columns={0: 'visitors_count'})
        tml_visitors_count = tml_visitors_count.sort_values(by="Дата")

        # Quick check of the aggregated data
        fig = plt.figure(figsize=(10, 4))
        sns.lineplot(data=tml_visitors_count, x="Дата", y="visitors_count")
        fig.savefig('src/output/plots/eda/visitors_count_ts_plot.png', bbox_inches="tight")
        plt.close()

        # First three values look odd (row 1 - is assumed to be test row, row 2 and 3 are from the venue opening
        # Thus these rows are considered outliers and are removed from the analysis
        tml_visitors_count = tml_visitors_count.iloc[3:]

        # Add the weekday column
        tml_visitors_count["weekday"] = tml_visitors_count['Дата'].dt.weekday + 1
        print("Check weekday row count : ")
        print(tml_visitors_count["weekday"].value_counts().sort_values())
        # From this it looks that the Monday is non working day for the TML.
        # There is only 1 observation that will be removed
        tml_visitors_count = tml_visitors_count[tml_visitors_count["weekday"] != 1]

        # Check for Gaps in the timeseries data
        tml_visitors_count['gap_in_days'] = tml_visitors_count['Дата'].sort_values().diff()
        tml_visitors_count['gap_fg'] = tml_visitors_count['Дата'].sort_values().diff() > pd.to_timedelta('1 day')

        print("Max gap is : ", tml_visitors_count['gap_in_days'].max())

        # Fix gaps (resample with interpolate : option spline)
        # spline: Estimates values that minimize overall curvature, thus obtaining a smooth surface passing
        # through the input points.
        tml_visitors_ts = tml_visitors_count[['Дата', 'visitors_count']]
        tml_visitors_ts.index = tml_visitors_ts['Дата']
        tml_visitors_ts = tml_visitors_ts[["visitors_count"]]
        tml_visitors_ts = tml_visitors_ts.resample('1D').mean().interpolate(option='spline')
        tml_visitors_ts = tml_visitors_ts.reset_index()
        tml_visitors_ts["weekday"] = tml_visitors_ts['Дата'].dt.weekday + 1

        # Note : Monday is again removed (off day for the TML)
        tml_visitors_ts = tml_visitors_ts[tml_visitors_ts["weekday"] != 1]

        # Add is_weekend flag
        tml_visitors_ts['is_weekend'] = tml_visitors_ts['weekday'].isin([6, 7]).astype('int')

        # Encode the day as sine/cosine (cyclical feature).
        # Hint divide by 6 because Monday is closed. Otherwise, we have 7 weekdays
        tml_visitors_ts['weekday_sin'] = np.sin(tml_visitors_ts['weekday'] * (2 * np.pi / 6))
        tml_visitors_ts['weekday_cos'] = np.cos(tml_visitors_ts['weekday'] * (2 * np.pi / 6))

        # Add month column and encode sine/cosine
        tml_visitors_ts["month"] = tml_visitors_ts['Дата'].dt.month
        cyclical = CyclicalFeatures(variables=None, drop_original=True)
        tml_visitors_ts[["month_sin", "month_cos"]] = cyclical.fit_transform(tml_visitors_ts[["month"]])

        # create new column that displays quarter from date column
        tml_visitors_ts['quarter'] = tml_visitors_ts['Дата'].dt.quarter

        # Plot the final time series
        fig = plt.figure(figsize=(10, 4))
        sns.lineplot(data=tml_visitors_count, x="Дата", y="visitors_count")
        fig.savefig('src/output/plots/eda/visitors_count_ts_plot_interpolated.png', bbox_inches="tight")
        plt.close()
        return tml_visitors_ts

    def prepare_school_holidays(self):
        """
        Biggest share of visitors is among the range 7-14 years (kids). Thus school holidays, summer vacation and
        other will play a role.

        :return:
        """
        # from 30.06 to 14.09 we have summer vacation. Take the big range
        summer_holidays_y1 = pd.DataFrame(pd.date_range(start='2018-06-30', end='2018-09-14'))
        summer_holidays_y1 = summer_holidays_y1.rename(columns={0: 'Дата'})
        summer_holidays_y1["is_summer_holiday"] = 1

        summer_holidays_y2 = pd.DataFrame(pd.date_range(start='2019-06-30', end='2019-09-14'))
        summer_holidays_y2 = summer_holidays_y2.rename(columns={0: 'Дата'})
        summer_holidays_y2["is_summer_holiday"] = 1

        school_holidays = summer_holidays_y1.append(summer_holidays_y2)
        return school_holidays

    def prepare_weather_data(self):
        """
        Read, combine and select weather features to use for modeling

        * Dew point
            the higher the dew point the muggier (https://www.youtube.com/watch?v=Cuf12bYTHHs)

        :return:
        """
        column_names = [
            "day_of_week",
            "Temperature (°F)_Max",
            "Temperature (°F)_Avg",
            "Temperature (°F)_Min",
            "Dew Point (°F)_Max",
            "Dew Point (°F)_Avg",
            "Dew Point (°F)_Min",
            "Humidity (%)_Max",
            "Humidity (%)_Avg",
            "Humidity (%)_Min",
            "Wind Speed (mph)_Max",
            "Wind Speed (mph)_Avg",
            "Wind Speed (mph)_Min",
            "Pressure (in)_Max",
            "Pressure (in)_Avg",
            "Pressure (in)_Min",
            "Precipitation (in)",
            "Дата"
        ]

        def read_weather(path: str, date_start: str, date_end: str):
            df = pd.read_excel(io=path, skiprows=2, sheet_name=None, header=None)
            df_all = pd.concat(df.values(), ignore_index=True)
            df_all.dropna(how="any", inplace=True)
            df_all = df_all.reset_index(drop=True)
            df_all["Дата"] = pd.date_range(start=date_start, end=date_end, freq='1D').date
            return df_all

        # Weather
        we_2018 = read_weather(path="src/resources/weather data 2018.xlsx", date_start='2018-04-01',
                               date_end='2018-12-31')
        we_2019 = read_weather(path="src/resources/weather data 2019.xlsx", date_start='2019-01-01',
                               date_end='2019-12-31')
        we_2020 = read_weather(path="src/resources/weather data 2020.xlsx", date_start='2020-01-01',
                               date_end='2020-03-31')

        weather_df = pd.concat([we_2018, we_2019, we_2020], ignore_index=True)
        weather_df.columns = column_names

        # run profiling (run once)
        # profile = ProfileReport(weather_df, title="Profiling Report")
        # profile.to_file(output_file=f'src/output/profiling/weather_data_all_years.html')

        # Findings from the profiling :
        # let's focus for now the average values
        # tmp_avg is highly correlated with dew_point and Humidity_avg - thus we can omit them for the time being
        # Assumption is that we are going to train OLS model
        # Precipitation column is uniform thus removed

        focus_fields = ["Дата", "Temperature (°F)_Avg", "Wind Speed (mph)_Avg", "Pressure (in)_Avg"]
        weather_df = weather_df[focus_fields]

        # convert Fahrenhait to Celsius
        weather_df["temperature_celsius_avg"] = (weather_df["Temperature (°F)_Avg"] - 32) * 5 / 9

        # convert mph to kph
        weather_df["wind_speed_kph_avg"] = weather_df["Wind Speed (mph)_Avg"] * 1.60934

        weather_df = weather_df.rename(columns={'Pressure (in)_Avg': 'pressure_in_avg'})

        return weather_df[["Дата", "temperature_celsius_avg", "wind_speed_kph_avg", "pressure_in_avg"]]

    def auto_correlation_plot(self):
        """
        Assess the effect of the past data over future using the Partial auto correlation plot

        :return:
        """

        acf = plot_acf(visitors_ts['visitors_count'], lags=25)
        pacf = plot_pacf(visitors_ts['visitors_count'], lags=25)

        #TODO: consider if we need to do other things