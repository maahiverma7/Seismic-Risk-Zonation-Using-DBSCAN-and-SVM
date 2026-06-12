import requests
import pandas as pd

def extract_india_seismic_data():
    print("Connecting to USGS API...")
    
    # The official USGS API endpoint
    url = "https://earthquake.usgs.gov/fdsnws/event/1/query"
    
    # API Parameters: India + Neighboring Fault Lines
    params = {
        "format": "geojson",
        "starttime": "2014-01-01",  # 10 years of solid historical data
        "endtime": "2024-01-01",
        "minmagnitude": 3.5,        # 3.5+ removes micro-tremors, keeps real threats
        "minlatitude": 6.0,         # Bottom: Below Sri Lanka & Andaman
        "maxlatitude": 38.0,        # Top: Above Kashmir, includes Nepal/Tibet
        "minlongitude": 68.0,       # Left: Includes Pakistan fault lines
        "maxlongitude": 98.0        # Right: Includes Myanmar/Bangladesh borders
    }
    
    print("Downloading 10 years of earthquake data (This might take 10-20 seconds)...")
    response = requests.get(url, params=params)
    
    if response.status_code != 200:
        print("Error pulling data. USGS might be busy.")
        return
        
    data = response.json()
    
    # Parse the extracted JSON into a clean list
    earthquakes = []
    for feature in data['features']:
        props = feature['properties']
        coords = feature['geometry']['coordinates']
        
        earthquakes.append({
            "place": props['place'],
            "magnitude": props['mag'],
            "longitude": coords[0],
            "latitude": coords[1],
            "depth_km": coords[2],
            "time": pd.to_datetime(props['time'], unit='ms')
        })
        
    # Convert to a clean DataFrame
    df = pd.DataFrame(earthquakes)
    
    # Save the extracted data
    filename = "India_Subcontinent_Seismic_Data_10_Years.csv"
    df.to_csv(filename, index=False)
    
    print("--------------------------------------------------")
    print("SUCCESS! Data extracted and cleaned.")
    print(f"Total Earthquakes Found: {len(df)}")
    print(f"File Saved As: {filename}")
    print("Ready to send to the ML team!")
    print("--------------------------------------------------")

# Run the tool
extract_india_seismic_data()