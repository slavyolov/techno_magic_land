import pandas as pd
import seaborn as sns
from pandas_profiling import ProfileReport
import matplotlib.pyplot as plt
import numpy as np
from feature_engine.creation import CyclicalFeatures
from datetime import datetime


class DataPreparation:
    def __init__(self, config):
        self.config = config

    def run(self):
        """
        Execute the data processing stages

        :return:
        """
        tables = self.read_data()

        visitors_ts = self.prepare_tml_visitors_count(tables=tables, fill_gaps=True) #TODO: ddecide if true or false
        summer_holidays, winter_holidays, spring_holidays, school_exams, school_holidays = self.prepare_school_holidays()
        weather_df = self.prepare_weather_data()

        # combine source tables
        combined_df = pd.merge(visitors_ts, weather_df, left_on="Дата", right_on="Дата", how="inner")
        combined_df = pd.merge(combined_df, summer_holidays, left_on="Дата", right_on="Дата", how="left")
        combined_df = pd.merge(combined_df, winter_holidays, left_on="Дата", right_on="Дата", how="left")
        combined_df = pd.merge(combined_df, spring_holidays, left_on="Дата", right_on="Дата", how="left")
        combined_df = pd.merge(combined_df, school_exams, left_on="Дата", right_on="Дата", how="left")
        combined_df = pd.merge(combined_df, school_holidays, left_on="Дата", right_on="Дата", how="left")

        # fill in the missing public holidays in the school holidays column
        combined_df["is_school_holiday"] = np.where(combined_df["is_public_holiday"] == 1, 1,
                                                    combined_df["is_school_holiday"])

        combined_df["is_summer_holiday"].fillna(0, inplace=True)
        combined_df["is_winter_holiday"].fillna(0, inplace=True)
        combined_df["is_spring_holiday"].fillna(0, inplace=True)
        combined_df["school_exams"].fillna(0, inplace=True)
        combined_df["is_school_holiday"].fillna(0, inplace=True)

        # get school day :
        combined_df['school_day'] = np.where((combined_df["is_weekend"] == 0) &
                                             (combined_df["is_school_holiday"] == 0), 1, 0)

        return tables, visitors_ts, school_holidays, combined_df

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

    def prepare_tml_visitors_count(self, tables, fill_gaps=False):
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

        # Add the day_of_week column
        tml_visitors_count["day_of_week"] = tml_visitors_count['Дата'].dt.day_of_week + 1
        print("Check day_of_week row count : ")
        print(tml_visitors_count["day_of_week"].value_counts().sort_values())
        # From this it looks that the Monday is non working day for the TML.
        # There is only 1 observation that will be removed
        tml_visitors_count = tml_visitors_count[tml_visitors_count["day_of_week"] != 1]

        if fill_gaps:
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
            tml_visitors_ts["day_of_week"] = tml_visitors_ts['Дата'].dt.day_of_week + 1

            # Note : Monday is again removed (off day for the TML)
            tml_visitors_ts = tml_visitors_ts[tml_visitors_ts["day_of_week"] != 1]
        else:
            tml_visitors_ts = tml_visitors_count.copy(deep=True)

        # Lag y-1
        tml_visitors_ts["visitors_count_lag_1"] = tml_visitors_ts["visitors_count"].shift(1)  # вчера
        tml_visitors_ts["visitors_count_lag_2"] = tml_visitors_ts["visitors_count"].shift(2)
        tml_visitors_ts["visitors_count_lag_3"] = tml_visitors_ts["visitors_count"].shift(3)
        tml_visitors_ts["visitors_count_lag_7"] = tml_visitors_ts["visitors_count"].shift(7)  # преди 7 дни

        tml_visitors_ts["visitors_count_lag_5"] = tml_visitors_ts["visitors_count"].shift(5)  # преди 7 дни
        tml_visitors_ts["visitors_count_lag_9"] = tml_visitors_ts["visitors_count"].shift(9)  # преди 7 дни
        tml_visitors_ts["visitors_count_lag_10"] = tml_visitors_ts["visitors_count"].shift(10)  # преди 7 дни
        tml_visitors_ts["visitors_count_lag_11"] = tml_visitors_ts["visitors_count"].shift(11)  # преди 7 дни

        # day_of_week mean (last 4 weeks mean)
        tml_day_of_week_rolling = tml_visitors_ts.copy(deep=True)
        tml_day_of_week_rolling.index = tml_day_of_week_rolling['Дата']

        tml_day_of_week_rolling = tml_day_of_week_rolling.groupby("day_of_week").rolling(4).agg(
            {'visitors_count': [np.mean, np.min, np.max, np.median, np.std]})

        tml_day_of_week_rolling.columns = ["_".join(x) for x in tml_day_of_week_rolling.columns.ravel()]
        tml_day_of_week_rolling = tml_day_of_week_rolling.reset_index()
        tml_day_of_week_rolling = tml_day_of_week_rolling.drop('day_of_week', axis=1)
        tml_day_of_week_rolling.columns = ['Дата', 'visitors_count_mean', 'visitors_count_min',
                                           'visitors_count_max', 'visitors_count_median', 'visitors_count_std']


        # Combine the day_of_week mean to the main data frame
        tml_visitors_ts = pd.merge(tml_visitors_ts, tml_day_of_week_rolling, on="Дата", how="inner")

        # Add is_weekend flag
        tml_visitors_ts['is_weekend'] = tml_visitors_ts['day_of_week'].isin([6, 7]).astype('int')

        # Encode the day as sine/cosine (cyclical feature).
        # Hint divide by 6 because Monday is closed. Otherwise, we have 7 day_of_weeks
        tml_visitors_ts['day_of_week_sin'] = np.sin(tml_visitors_ts['day_of_week'] * (2 * np.pi / 6))
        tml_visitors_ts['day_of_week_cos'] = np.cos(tml_visitors_ts['day_of_week'] * (2 * np.pi / 6))

        # Add month column and encode sine/cosine
        tml_visitors_ts["month"] = tml_visitors_ts['Дата'].dt.month
        cyclical = CyclicalFeatures(variables=None, drop_original=True)
        tml_visitors_ts[["month_sin", "month_cos"]] = cyclical.fit_transform(tml_visitors_ts[["month"]])

        # create new column that displays quarter from date column
        tml_visitors_ts['quarter'] = tml_visitors_ts['Дата'].dt.quarter

        # one hot encoding weekday
        tml_visitors_ts = pd.concat([tml_visitors_ts,
                   pd.get_dummies(tml_visitors_ts['day_of_week'], drop_first=True, prefix="weekday")], axis=1)

        # one hot encoding season
        conditions = [
            (tml_visitors_ts['month'].isin([12, 1, 2])),
            (tml_visitors_ts['month'].isin([3, 4, 5])),
            (tml_visitors_ts['month'].isin([6, 7, 8])),
            (tml_visitors_ts['month'].isin([9, 10, 11])),
        ]

        # create a list of the values we want to assign for each condition
        values = ['winter', 'spring', 'summer', 'autumn']

        # create a new column and use np.select to assign values to it using our lists as arguments
        tml_visitors_ts['season'] = np.select(conditions, values)

        tml_visitors_ts = pd.concat([tml_visitors_ts,
                   pd.get_dummies(tml_visitors_ts['season'])], axis=1)

        # drop the winter column
        tml_visitors_ts = tml_visitors_ts.drop('winter', axis=1)

        # add public holidays
        tml_visitors_ts["is_public_holiday"] = np.where(tml_visitors_ts["Дата"].isin(self.config.public_holidays), 1, 0)

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
        summer_holidays_y2 = pd.DataFrame(pd.date_range(start='2019-06-30', end='2019-09-14'))
        summer_holidays = summer_holidays_y1.append(summer_holidays_y2)
        summer_holidays = summer_holidays.rename(columns={0: 'Дата'})
        summer_holidays["is_summer_holiday"] = 1

        # Winter holidays :
        winter_h_y1 = pd.DataFrame(pd.date_range(start='2018-12-22', end='2019-01-02'))
        winter_h_y2 = pd.DataFrame(pd.date_range(start='2019-12-21', end='2020-01-05'))
        winter_holidays = winter_h_y1.append(winter_h_y2)
        winter_holidays = winter_holidays.rename(columns={0: 'Дата'})
        winter_holidays["is_winter_holiday"] = 1

        # Spring school holidays
        spring_holidays = pd.DataFrame(pd.date_range(start='2019-03-30', end='2019-04-04'))
        spring_holidays = spring_holidays.rename(columns={0: 'Дата'})
        spring_holidays["is_spring_holiday"] = 1

        # Other holidays
        school_exams = pd.DataFrame({"Дата": [datetime(2019, 5, 9), datetime(2019, 5, 10), datetime(2019, 5, 14),
                                              datetime(2019, 5, 16)],
                                     "school_exams": [1, 1, 1, 1]})

        # combine all holidays
        school_holidays = pd.DataFrame(
            summer_holidays["Дата"]
            .append(school_exams["Дата"])
            .append(spring_holidays["Дата"])
            .append(winter_holidays["Дата"])
            .reset_index(drop=True)
            .sort_values()
                                       )

        school_holidays["is_school_holiday"] = 1

        return summer_holidays, winter_holidays, spring_holidays, school_exams, school_holidays

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
            df_all["Дата"] = pd.date_range(start=date_start, end=date_end, freq='1D') #.date # uncomment if you want the date
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
        # let's focus for now the average values; Second iteration focus on MAX values
        # tmp_avg is highly correlated with dew_point and Humidity_avg - thus we can omit them for the time being
        # Assumption is that we are going to train OLS model
        # Precipitation column is uniform thus removed

        focus_fields = ["Дата", "Temperature (°F)_Max", "Wind Speed (mph)_Max", "Pressure (in)_Max"]
        weather_df = weather_df[focus_fields]

        # convert Fahrenhait to Celsius
        weather_df["temperature_celsius_max"] = (weather_df["Temperature (°F)_Max"] - 32) * 5 / 9

        # convert mph to kph
        weather_df["wind_speed_kph_max"] = weather_df["Wind Speed (mph)_Max"] * 1.60934

        weather_df = weather_df.rename(columns={'Pressure (in)_Max': 'pressure_in_max'})

        # create a list of the values we want to assign for each condition
        conditions = [
            (weather_df['temperature_celsius_max'].between(34, 42)),
            (weather_df['temperature_celsius_max'].between(26, 33.99)),
            (weather_df['temperature_celsius_max'].between(20, 25.99)),
            (weather_df['temperature_celsius_max'].between(10, 19.99)),
            (weather_df['temperature_celsius_max'].between(0, 9.99)),
            (weather_df['temperature_celsius_max'].between(-20, -0.01)),
        ]

        values = ['very hot', 'hot', 'warm', 'cool', 'cold', 'freezing']

        # create a new column and use np.select to assign values to it using our lists as arguments
        weather_df['weather_classification'] = np.select(conditions, values)
        weather_df = pd.concat([weather_df, pd.get_dummies(weather_df['weather_classification'])], axis=1)
        weather_df = weather_df.drop('freezing', axis=1)

        return weather_df[["Дата", "temperature_celsius_max", "wind_speed_kph_max", "pressure_in_max",
                           'hot', 'warm', 'cool', 'cold'
                           ]]
