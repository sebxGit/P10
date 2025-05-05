from meteostat import Stations, Hourly, Point
from datetime import datetime

def fetch_denver_weather(start, end):
    data = Hourly(
        loc=Point(39.7833, -104.8667), # Denver, CO
        start=start,
        end=end
    )
    data = data.fetch()

    data = data[['temp']].rename(columns={'temp': 'temperature'})

    print(data.head())  
    print(data.tail())  # Print the last few rows of the data

    # Save to CSV
    data.to_csv('denver_weather.csv', index=True)

    return data


if __name__ == '__main__':
    start_date = datetime(2021, 5, 30)  # Pass year, month, day as integers
    end_date = datetime(2023, 5, 30)    # Pass year, month, day as integers
    weather = fetch_denver_weather(start=start_date, end=end_date)






