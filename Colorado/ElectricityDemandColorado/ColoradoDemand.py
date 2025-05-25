import pandas as pd
import re
import matplotlib.pyplot as plt

files = ['Jan2023.csv', 'Feb2023.csv', 'Marts2023.csv', 'May2023.csv']

# Columns to keep
columns_to_keep = ["BA Code", "Timestamp (Hour Ending)", "Demand (MWh)"]

# Read and concatenate all files
df = pd.concat([pd.read_csv(f, usecols=lambda c: c in columns_to_keep)
               for f in files], ignore_index=True)

# PLot the data
plt.figure(figsize=(10, 5))
plt.plot(df["Timestamp (Hour Ending)"], df["Demand (MWh)"])
plt.title('Demand Over Time')
plt.xlabel('Timestamp (Hour Ending)')
plt.ylabel('Demand (MWh)')

# Adjust x-ticks to show fewer labels for better spacing
plt.xticks(ticks=range(0, len(df), len(df)//10))  # Show 10 evenly spaced ticks

plt.tight_layout()
plt.show()

# Save to a new CSV
df.to_csv('ColoradoConsumption.csv', index=False)
