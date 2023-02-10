# techno_magic_land

# Code steps :
1. main.py takes care for :
    - Data Profiling / Exporatory data analysis
    - Feature engineering
    - Outlier removal
    - Modeling using OLS

# Details :
- Data Preparation :
  - Weather data :
    - selecting non correlated features
    - Creating new features based on the temperature binning:
      - ['very hot', 'hot', 'warm', 'cool', 'cold', 'freezing']
  - Holidays data created by using
    - public holidays (https://www.officeholidays.com/countries/bulgaria/2019)
    - school holidays/ school days (https://ucha.se/motiviramse/kalendar-za-uchebnata-2018-2019-godina/)
  - Features extracted from the TML dataset :
    - calendar features (weekend, day_of_week, month, season, and other)
    - lag features (e.g. y-1; y-10)
    - Aggregated lag features (e.g. median/mean on day_of_week level for the last 4 weeks)
- Output files :
  - src/output/plots/ : Diagrams created through the analysis
  - src/output/model : OLS summary files
  - src/output/data : data files
  - src/output/profiling : tables profiling