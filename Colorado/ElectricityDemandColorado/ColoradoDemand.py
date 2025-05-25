import pandas as pd
import re
import matplotlib.pyplot as plt
from itertools import groupby
from operator import itemgetter

files = ['Jan2023.csv', 'Feb2023.csv', 'Marts2023.csv', 'May2023.csv']

# Columns to keep
columns_to_keep = ["BA Code", "Timestamp (Hour Ending)", "Demand (MWh)"]

# Read and concatenate all files
df = pd.concat([pd.read_csv(f, usecols=lambda c: c in columns_to_keep)
               for f in files], ignore_index=True)

df2 = df.copy()
df2['Timestamp (Hour Ending)'] = df2['Timestamp (Hour Ending)'].str.replace(r'\s\w{3}$', '', regex=True)

# Convert to datetime
df2['Timestamp (Hour Ending)'] = pd.to_datetime(df2['Timestamp (Hour Ending)'])

full_date_range = pd.date_range(start=df2['Timestamp (Hour Ending)'].min(), 
                                end=df2['Timestamp (Hour Ending)'].max(), 
                                freq='H')

present_dates = set(df2['Timestamp (Hour Ending)'])

missing_dates = missing_dates = sorted([timestamp for timestamp in full_date_range if timestamp not in present_dates])

formatted_missing_dates = [date.strftime('%m-%d-%Y %H:%M') for date in missing_dates]

missing_data_rows = sorted(list(df2[df2['Demand (MWh)'].isnull() | (df2['Demand (MWh)'] == '')]['Timestamp (Hour Ending)']))
formatted_missing_data_rows = [date.strftime('%m-%d-%Y %H:%M') for date in missing_data_rows]

formatted_missing_dates.remove('03-12-2023 02:00')

merged_and_sorted = sorted(set(formatted_missing_dates + formatted_missing_data_rows))

print(df2['Timestamp (Hour Ending)'].min(), "to", df2['Timestamp (Hour Ending)'].max())

print("Missing Dates and Data Rows:")
print(merged_and_sorted)


# PLot the data
plt.figure(figsize=(10, 5))
plt.plot(df["Timestamp (Hour Ending)"], df["Demand (MWh)"])
plt.title('Demand Over Time')
plt.xlabel('Timestamp (Hour Ending)')
plt.ylabel('Demand (MWh)')
plt.xticks(rotation=45)

# Adjust x-ticks to show fewer labels for better spacing
plt.xticks(ticks=range(0, len(df), len(df)//10))  # Show 10 evenly spaced ticks

plt.tight_layout()
plt.show()

# Save to a new CSV
df.to_csv('ColoradoConsumption.csv', index=False)
