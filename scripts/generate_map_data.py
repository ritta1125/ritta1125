import geopandas as gpd
import json
import os

def generate_korea_topojson():
    # Create the output directory if it doesn't exist
    os.makedirs('static/data', exist_ok=True)
    
    # Load South Korea administrative boundaries
    url = "https://raw.githubusercontent.com/southkorea/southkorea-maps/master/kostat/2018/json/skorea-municipalities-2018-topo.json"
    
    # Download and save the file
    import requests
    response = requests.get(url)
    
    if response.status_code == 200:
        with open('static/data/skorea-municipalities-2018-topo.json', 'w', encoding='utf-8') as f:
            f.write(response.text)
        print("Successfully downloaded and saved Korea map data")
    else:
        print(f"Failed to download map data: {response.status_code}")

if __name__ == "__main__":
    generate_korea_topojson() 