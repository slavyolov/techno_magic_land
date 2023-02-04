import pandas as pd
from pandas_profiling import ProfileReport


class DataPreparation:
    def __init__(self, config):
        self.config = config

    def run(self):
        """
        Execute the data processing stages

        :return:
        """
        tables = self.read_data()

        # Handle missing data for the 1st file
        tml_visitors_p1 = tables["tml_visitors_p1"].copy(deep=True)

        tml_visitors_p1["Точки за посещението"].fillna(0, inplace=True)
        tml_visitors_p1.loc[tml_visitors_p1['Дата на излизане'].isnull(),
                            'Дата на излизане'] = tml_visitors_p1['Дата на влизане']

        # Aggregate on day level to get the intensity (Интензивност на посещения - брой посетители на дневна база)
        tml_visitors_p1.groupby(by=["Дата"])[]

        return tables

    def read_data(self):
        xls = pd.ExcelFile('src/resources/TML_vistors_case_study_data.xlsx')

        df1 = pd.read_excel(xls, 'Таблица 1')
        df2 = pd.read_excel(xls, 'Таблица 2')
        df3 = pd.read_excel(xls, 'Таблица 3')

        we_2018 = pd.read_excel(io="src/resources/weather data 2018.xlsx", header=1)
        we_2019 = pd.read_excel(io="src/resources/weather data 2019.xlsx", header=1)
        we_2020 = pd.read_excel(io="src/resources/weather data 2020.xlsx", header=1)

        return {
            'tml_visitors_p1': df1,
            'tml_visitors_p2': df2,
            'tml_visitors_p3': df3,
            'we_2018': we_2018,
            'we_2019': we_2019,
            'we_2020': we_2020,
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

        profile = ProfileReport(tables["we_2018"], title="Profiling Report")
        profile.to_file(output_file=f'src/output/profiling/we_2018.html')

        profile = ProfileReport(tables["we_2019"], title="Profiling Report")
        profile.to_file(output_file=f'src/output/profiling/we_2019.html')

        profile = ProfileReport(tables["we_2020"], title="Profiling Report")
        profile.to_file(output_file=f'src/output/profiling/we_2020.html')

        print("Profiling done!")
