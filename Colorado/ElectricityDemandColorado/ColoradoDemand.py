import pandas as pd
import matplotlib.pyplot as plt

files = ['Jan2023.csv', 'Feb2023.csv', 'Marts2023.csv', 'April2023.csv', 'May2023.csv']

# Columns to keep
columns_to_keep = ["Timestamp (Hour Ending)", "Demand (MWh)"]

# Read and concatenate all files
df = pd.concat([pd.read_csv(f, usecols=lambda c: c in columns_to_keep)
               for f in files], ignore_index=True)

df = df.drop_duplicates()

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


#cuts
# 02-04-2023 00:00 to 02-05-2023 01:00
# 02-28-2023 00:00 to 03-05-2023 01:00
range1_start = pd.Timestamp('2023-02-04 00:00')
range1_end = pd.Timestamp('2023-02-05 01:00')
range2_start = pd.Timestamp('2023-02-28 00:00')
range2_end = pd.Timestamp('2023-03-05 01:00')
start_date = pd.Timestamp('2023-01-04 00:00')
end_date = pd.Timestamp('2023-05-22 23:00')

df['Timestamp (Hour Ending)'] = pd.to_datetime(df['Timestamp (Hour Ending)'])

# df_part1 = df[df['Timestamp (Hour Ending)'] < range1_start & (df['Timestamp (Hour Ending)'] >= start_date)]
# df_part2 = df[(df['Timestamp (Hour Ending)'] >= range1_end) & (df['Timestamp (Hour Ending)'] <= range2_start)]
# df_part3 = df[df['Timestamp (Hour Ending)'] > range2_end & (df['Timestamp (Hour Ending)'] <= end_date)]

# df_part1.to_csv('ColoradoDemand_Part1.csv', index=False)
# df_part2.to_csv('ColoradoDemand_Part2.csv', index=False)
# df_part3.to_csv('ColoradoDemand_Part3.csv', index=False)

df_part1 = df[(df['Timestamp (Hour Ending)'] < range1_start) & (df['Timestamp (Hour Ending)'] >= start_date)]
df_part2 = df[(df['Timestamp (Hour Ending)'] >= range1_end) & (df['Timestamp (Hour Ending)'] <= range2_start)]
df_part3 = df[(df['Timestamp (Hour Ending)'] >= range2_end) & (df['Timestamp (Hour Ending)'] <= end_date)]

df_part1.to_csv('ColoradoDemand_Part1.csv', index=False)
df_part2.to_csv('ColoradoDemand_Part2.csv', index=False)
df_part3.to_csv('ColoradoDemand_Part3.csv', index=False)

print(df_part1.shape, df_part2.shape, df_part3.shape)
print(df.shape)




print(df_part1.shape, df_part2.shape, df_part3.shape)
print(df.shape)

# PLot the data
# plt.figure(figsize=(10, 5))
# plt.plot(df["Timestamp (Hour Ending)"], df["Demand (MWh)"])
# plt.title('Demand Over Time')
# plt.xlabel('Timestamp (Hour Ending)')
# plt.ylabel('Demand (MWh)')
# plt.xticks(rotation=45)

# # Adjust x-ticks to show fewer labels for better spacing
# plt.xticks(ticks=range(0, len(df), len(df)//10))  # Show 10 evenly spaced ticks

# plt.tight_layout()
# plt.show()

# # Save to a new CSV
# df.to_csv('ColoradoConsumption.csv', index=False)


# Ensure the 'Timestamp (Hour Ending)' column is in datetime format
df['Timestamp (Hour Ending)'] = pd.to_datetime(df['Timestamp (Hour Ending)'])

# Create a complete range of timestamps based on the min and max timestamps in the dataset
full_range = pd.date_range(start=df['Timestamp (Hour Ending)'].min(),
                           end=df['Timestamp (Hour Ending)'].max(),
                           freq='H')  # Hourly frequency

# Reindex the DataFrame to include the full range of timestamps
df = df.set_index('Timestamp (Hour Ending)').reindex(full_range).reset_index()

# Rename the index column back to 'Timestamp (Hour Ending)'
df.rename(columns={'index': 'Timestamp (Hour Ending)'}, inplace=True)

df.to_csv('ColoradoDemand_Full.csv', index=False)

# Plot the data
plt.figure(figsize=(10, 5))
plt.plot(df["Timestamp (Hour Ending)"], df["Demand (MWh)"],
         label='Demand (MWh)', color='blue')
plt.title('Demand Over Time (Including Missing Windows)')
plt.xlabel('Timestamp (Hour Ending)')
plt.ylabel('Demand (MWh)')
plt.xticks(rotation=45)

# Add a legend
plt.legend()

# Adjust layout for better spacing
plt.tight_layout()

plt.savefig('ColoradoDemand_Full.png')

# Show the plot
plt.show()
