df['week'] =  df.index.isocalendar().week
df['weekday']= df.index.dayofweek
df['hour'] = df.index.hour
df['date']=df.index.date
df['year']=df.index.year
df['month'] = df.index.month
Dk_holidays = holidays.country_holidays('DK', years=range(2020,2026))
DE_holidays = holidays.country_holidays('DE', years=range(2020,2026))
#label business day and holidays
def get_peak_status(hour):
    if 7 <= hour <= 21:
        return 1
    else:
        return 0

df['peak_status'] = df['hour'].apply(get_peak_status)

# Identify peak and off-peak
df['date'] = pd.to_datetime(df['date']).dt.date
def get_businessday(date):
    if date in Dk_holidays:
        return 0
    # Check if date is Saturday (5) or Sunday (6)
    elif date.weekday() in (5,6):
        return 0
    else:
        return 1

df['businessday'] = df['date'].apply(get_businessday)

def get_DE_holidays(date):
    # make sure 'date' is a Python date object
    d = date.date() if hasattr(date, 'date') else date

    if d in DE_holidays and d not in Dk_holidays:
        return 0
    else:
        return 1

df['DE_holiday'] = df['date'].apply(get_DE_holidays)

def get_christmas_time(date):
    date = pd.to_datetime(date)
    month = date.month
    day = date.day

    if (month == 12 and day >= 24) or (month == 1 and day <= 6):
        return 1
    else:
        return 0
df['Xmas_period']= df['date'].apply(get_christmas_time)


month_map = {
    'Jan': 1, 'Feb': 2, 'Mar': 3, 'Apr': 4,
    'May': 5, 'Jun': 6, 'Jul': 7, 'Aug': 8,
    'Sep': 9, 'Oct': 10, 'Nov': 11, 'Dec': 12
}
df_gas.rename(columns={'Year': 'year', 'Month': 'month'}, inplace=True)
df_gas['month'] = df_gas['month'].map(month_map)

df_gas['month'] = df_gas['month'].astype(int)
df_gas['year'] = df_gas['year'].astype(int)

df = pd.merge(df, df_gas, on=['year', 'month'], how='left')

df_temp['week'] = df_temp.index.isocalendar().week
weekly_avg_temp = df_temp.groupby(['week'])[['temp_max', 'temp_min']].mean().reset_index()

df = df.merge(dm, on=['year', 'week'], how='left')

dm_de['year'] = dm_de['year'].astype(int)
dm_de['week'] = dm_de['week'].str.extract(r'(\d+)').astype(int)
df = df.merge(dm_de, on=['year', 'week'], how='left',suffixes=('', '_DE'))

#correcting first 3 days of year which was not included in first week of year (week=53)
df.loc[:72,'Min Total Load']=df['Min Total Load'][73]
df.loc[:72,'Max Total Load']=df['Max Total Load'][73]
df.loc[:72,'Min Total Load_DE']=df['Min Total Load'][73]
df.loc[:72,'Max Total Load_DE']=df['Min Total Load'][73]
df.loc[(df['month'] == 1) & (df['week'] != 1), 'week'] = 1

def add_fourier_terms(df, time_col='hour', period=24, K=2, prefix='hour'):
    """
    Adds Fourier sin/cos terms for periodic features.
    K controls how many harmonics to include (K=1..3 is typical).
    """
    df = df.copy()
    for k in range(1, K+1):
        df[f'{prefix}_sin_{k}'] = np.sin(2 * np.pi * k * df[time_col] / period)
        df[f'{prefix}_cos_{k}'] = np.cos(2 * np.pi * k * df[time_col] / period)
    return df

df = add_fourier_terms(df, 'hour', period=24, K=2, prefix='hour')
df = add_fourier_terms(df, 'weekday', period=7, K=2, prefix='weekday')


#generating something like TMY file as input of solar and wind forecast

import glob
import os

def read_all_years(folder):
    # Get all matching Excel files
    files = sorted(glob.glob(os.path.join(
        folder,
        "energy-charts_Public_net_electricity_generation_in_Denmark_in_*.xlsx"
    )))

    dfs = []

    for f in files:
        # Extract year (last 4 digits before .xlsx)
        year = int(os.path.basename(f).split("_")[-1].replace(".xlsx", ""))

        # Skip if not between 2015 and 2024
        if year < 2015 or year > 2024:
            continue

        # Read only required columns
        df_RE = pd.read_excel(
            f,
            skiprows=[0, 2],
            usecols=["Date (GMT+1)","Solar", "Wind onshore", "Wind offshore"]
        )

        # Add the year
        df_RE["year"] = year

        dfs.append(df_RE)
        
    if dfs:
        df_all = pd.concat(dfs, ignore_index=True)
    return df_all

df_weather= read_all_years("path")

for year in [2015,2025]:
    for col in ["Solar", "Wind onshore", "Wind offshore"]:
        df_weather[col] = df_weather.groupby("year")[col].transform(lambda x: x / x.max())

df_weather.set_index('Date (GMT+1)', inplace=True)
df_weather['month'] = df_weather.index.month
df_weather['date'] = df_weather.index.date

df_weather = df_weather[~((df_weather.index.month == 2) & (df_weather.index.day == 29))]
monthly_avg = df_weather.groupby('month')[["Solar", "Wind onshore", "Wind offshore"]].mean()

representative_months = []

for m in range(1, 13):  # loop through 12 calendar months
    # Extract all Januarys, Februarys, etc.
    month_df = df_weather[df_weather['month'] == m]
    
    # Compute mean per year for this month
    month_year_means = (
        month_df.groupby('year')[["Solar", "Wind onshore", "Wind offshore"]].mean()
    )

    # Compute distance from long-term average for that month
    diffs = ((month_year_means - monthly_avg.loc[m]) ** 2).sum(axis=1)
    
    # Pick year whose monthly average is closest to long-term mean
    best_year = diffs.idxmin()
    
    # Extract that specific month’s data
    rep_month = df_weather[(df_weather['year'] == best_year) & (df_weather['month'] == m)]
    
    representative_months.append(rep_month)

tmy_data = pd.concat(representative_months).sort_values(['month', 'date']).reset_index(drop=True)

df['total RE']= df['Wind offshore']+ df['Wind onshore']+ df['Solar']
