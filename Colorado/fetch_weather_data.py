from meteostat import Stations, Hourly, Point
from datetime import datetime

def fetch_weather(start, end, location):
    data = Hourly(
        loc=location, 
        start=start,
        end=end
    )
    data = data.fetch()

    data = data[['temp']].rename(columns={'temp': 'temperature'})

    print(data.head())  
    print(data.tail())  # Print the last few rows of the data

    # get the station name
    stations = Stations()
    stations = stations.nearby(lat=location._lat, lon=location._lon, radius=50)
    stations = stations.fetch()
    station_name = stations.iloc[0]['name'] if not stations.empty else 'Unknown Station'
    # Sanitize the station name
    station_name = station_name.strip().replace(" ", "_").replace("/", "-")

    print(f"Station Name: {station_name}")

    # Save to CSV
    data.to_csv(f'{station_name}_weather.csv', index=True)


    return data


if __name__ == '__main__':
    #fetch_weather(start=datetime(2021, 5, 30), end=datetime(2023, 5, 30), location=Point(39.7833, -104.8667))  # Denver, CO

    fetch_weather(start=datetime(2025, 1, 1), end=datetime(2031, 12, 31), location=Point(55.6759, 12.5655))  # Copenhagen, Denmark






