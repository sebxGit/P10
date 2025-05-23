import pandas as pd
import re
import matplotlib.pyplot as plt

files = ['Jan2023.csv', 'Feb2023.csv', 'Marts2023.csv', 'May2023.csv']

# Columns to keep
columns_to_keep = ["BA Code", "Timestamp (Hour Ending)", "Demand (MWh)"]

# Read and concatenate all files
df = pd.concat([pd.read_csv(f, usecols=lambda c: c in columns_to_keep)
               for f in files], ignore_index=True)


# def remove_tz_abbr(s):
#     # Remove trailing timezone abbreviations like 'MST' or 'MDT'
#     return re.sub(r'\s*(MDT|MST)$', '', str(s))


# # Remove timezone abbreviations
# df["Timestamp (Hour Ending)"] = df["Timestamp (Hour Ending)"].apply(
#     remove_tz_abbr)

# # Let pandas infer the datetime format (handles "3/1/2023 12 a.m." etc.)
# #df["Timestamp (Hour Ending)"] = pd.to_datetime(df["Timestamp (Hour Ending)"])

# df["Timestamp (Hour Ending)"] = df["Timestamp (Hour Ending)"].apply(
#     remove_tz_abbr)

# # Specify the format to avoid the warning
# df["Timestamp (Hour Ending)"] = pd.to_datetime(
#     df["Timestamp (Hour Ending)"], format="%m/%d/%Y %I %p"
# )

def remove_tz_abbr(s):
    # Remove trailing timezone abbreviations like 'MST' or 'MDT'
    return re.sub(r'\s*(MDT|MST)$', '', str(s))

def clean_ampm(s):
    # Replace 'a.m.'/'p.m.' with 'AM'/'PM'
    s = str(s).replace('a.m.', 'AM').replace('p.m.', 'PM')
    return s

# Remove timezone abbreviations
df["Timestamp (Hour Ending)"] = df["Timestamp (Hour Ending)"].apply(
    remove_tz_abbr)

# Replace a.m./p.m. with AM/PM
df["Timestamp (Hour Ending)"] = df["Timestamp (Hour Ending)"].apply(clean_ampm)

# Now parse with explicit format
df["Timestamp (Hour Ending)"] = pd.to_datetime(
    df["Timestamp (Hour Ending)"], format="%m/%d/%Y %I %p"
)

# Localize to US/Mountain and convert to UTC
df["Timestamp (Hour Ending)"] = df["Timestamp (Hour Ending)"].dt.tz_localize(
    'US/Mountain', ambiguous='NaT', nonexistent='shift_forward').dt.tz_convert('UTC')

# Fill empty values in numeric columns with the column average
# for col in ["Demand (MWh)"]:
#     df[col] = pd.to_numeric(df[col], errors='coerce')  # Ensure numeric
#     mean_val = round(df[col].mean(), 1)
#     df[col] = df[col].fillna(mean_val)

df = df.set_index("Timestamp (Hour Ending)")


for col in ["Demand (MWh)"]:
    df[col] = pd.to_numeric(df[col], errors='coerce')
    # Interpolate based on time index
    df[col] = df[col].interpolate(method='time')
    # If any NaNs remain (e.g., at the start/end), fill with mean
    mean_val = round(df[col].mean(skipna=True), 1)
    df[col] = df[col].fillna(mean_val)

df = df.reset_index()

# PLot the data
plt.figure(figsize=(10, 5))
plt.plot(df["Timestamp (Hour Ending)"], df["Demand (MWh)"])
plt.title('Demand Over Time')
plt.xlabel('Timestamp (Hour Ending)')
plt.ylabel('Demand (MWh)')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Save to a new CSV
df.to_csv('ColoradoConsumption.csv', index=False)
