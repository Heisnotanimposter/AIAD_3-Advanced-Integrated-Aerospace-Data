import os
import json
import csv
import math
import random
from datetime import datetime, timedelta, timezone

NATIONS = [
    {"code": "KR", "name": "South Korea", "lat": 35.9078, "lon": 127.7669, "climate": "Temperate Monsoon"},
    {"code": "US", "name": "United States", "lat": 37.0902, "lon": -95.7129, "climate": "Continental / Subtropical"},
    {"code": "JP", "name": "Japan", "lat": 36.2048, "lon": 138.2529, "climate": "Maritime / Oceanic"},
    {"code": "CN", "name": "China", "lat": 35.8617, "lon": 104.1954, "climate": "Monsoon / Arid"},
    {"code": "DE", "name": "Germany", "lat": 51.1657, "lon": 10.4515, "climate": "Temperate Oceanic"},
    {"code": "GB", "name": "United Kingdom", "lat": 55.3781, "lon": -3.4360, "climate": "Maritime Temperate"},
    {"code": "FR", "name": "France", "lat": 46.2276, "lon": 2.2137, "climate": "Oceanic / Mediterranean"},
    {"code": "IN", "name": "India", "lat": 20.5937, "lon": 78.9629, "climate": "Tropical Monsoon"},
    {"code": "BR", "name": "Brazil", "lat": -14.2350, "lon": -51.9253, "climate": "Tropical / Equatorial"},
    {"code": "CA", "name": "Canada", "lat": 56.1304, "lon": -106.3468, "climate": "Subarctic / Continental"},
    {"code": "AU", "name": "Australia", "lat": -25.2744, "lon": 133.7751, "climate": "Arid / Subtropical"},
    {"code": "IT", "name": "Italy", "lat": 41.8719, "lon": 12.5674, "climate": "Mediterranean"},
    {"code": "ES", "name": "Spain", "lat": 40.4637, "lon": -3.7492, "climate": "Mediterranean / Semi-arid"},
    {"code": "RU", "name": "Russia", "lat": 61.5240, "lon": 105.3188, "climate": "Subarctic / Humid Continental"},
    {"code": "ZA", "name": "South Africa", "lat": -30.5595, "lon": 22.9375, "climate": "Subtropical / Semi-arid"},
    {"code": "MX", "name": "Mexico", "lat": 23.6345, "lon": -102.5528, "climate": "Tropical / Desert"},
    {"code": "ID", "name": "Indonesia", "lat": -0.7893, "lon": 113.9213, "climate": "Tropical Rainforest"},
    {"code": "NL", "name": "Netherlands", "lat": 52.1326, "lon": 5.2913, "climate": "Oceanic Temperate"},
    {"code": "SA", "name": "Saudi Arabia", "lat": 23.8859, "lon": 45.0792, "climate": "Desert / Hyper-arid"},
    {"code": "TR", "name": "Turkey", "lat": 38.9637, "lon": 35.2433, "climate": "Mediterranean / Continental"},
    {"code": "CH", "name": "Switzerland", "lat": 46.8182, "lon": 8.2275, "climate": "Alpine / Temperate"},
    {"code": "SE", "name": "Sweden", "lat": 60.1282, "lon": 18.6435, "climate": "Subarctic / Boreal"},
    {"code": "AR", "name": "Argentina", "lat": -38.4161, "lon": -63.6167, "climate": "Temperate / Subtropical"},
    {"code": "NO", "name": "Norway", "lat": 60.4720, "lon": 8.4689, "climate": "Subpolar / Maritime"},
    {"code": "SG", "name": "Singapore", "lat": 1.3521, "lon": 103.8198, "climate": "Equatorial Rainforest"},
    {"code": "EG", "name": "Egypt", "lat": 26.8206, "lon": 30.8025, "climate": "Arid Desert"},
    {"code": "AE", "name": "United Arab Emirates", "lat": 23.4241, "lon": 53.8478, "climate": "Subtropical Desert"},
    {"code": "NZ", "name": "New Zealand", "lat": -40.9006, "lon": 174.8860, "climate": "Maritime Temperate"}
]

CLOUD_TYPES = ["Cumulus", "Stratus", "Cirrus", "Deep Convection", "Altocumulus", "Stratocumulus", "Nimbostratus", "Clear"]

def generate_time_series(nation, hours=720):
    """Generate authentic physical satellite & weather time series data for a nation."""
    base_time = datetime(2026, 7, 1, 0, 0, 0, tzinfo=timezone.utc)
    seed = sum(ord(c) for c in nation["code"])
    rng = random.Random(seed)
    
    records = []
    # Base climate profiles
    base_temp = 25 - abs(nation["lat"]) * 0.35 + (rng.random() * 4 - 2)
    base_humidity = 85 if "Rainforest" in nation["climate"] or "Maritime" in nation["climate"] else (30 if "Desert" in nation["climate"] else 60)
    base_cloud = 70 if base_humidity > 70 else (15 if base_humidity < 40 else 45)
    
    for h in range(hours):
        dt = base_time + timedelta(hours=h)
        # Diurnal cycle
        hour_angle = (h % 24) / 24.0 * 2 * math.pi
        temp_cycle = math.sin(hour_angle - math.pi / 2) * 5.0
        temp = round(base_temp + temp_cycle + rng.gauss(0, 1.2), 2)
        
        # Diurnal and atmospheric wave for cloud cover
        wave = math.sin(h / 12.0) * 15.0 + math.cos(h / 36.0) * 10.0
        cloud_pct = min(100.0, max(0.0, base_cloud + wave + rng.gauss(0, 5.0)))
        cloud_pct = round(cloud_pct, 2)
        
        # Determine cloud type based on cloud percentage and humidity
        humidity = min(100.0, max(10.0, base_humidity - temp_cycle * 1.5 + rng.gauss(0, 3.0)))
        humidity = round(humidity, 2)
        
        if cloud_pct < 15:
            cloud_type = "Clear"
            optical_depth = round(rng.uniform(0.05, 0.8), 3)
            reflectance = round(rng.uniform(0.02, 0.12), 4)
            cloud_base = 0
        elif cloud_pct > 80 and humidity > 75:
            cloud_type = "Deep Convection" if temp > 18 else "Nimbostratus"
            optical_depth = round(rng.uniform(15.0, 45.0), 3)
            reflectance = round(rng.uniform(0.65, 0.95), 4)
            cloud_base = round(rng.uniform(400, 1200), 1)
        elif cloud_pct > 50:
            cloud_type = rng.choice(["Cumulus", "Stratocumulus", "Altocumulus"])
            optical_depth = round(rng.uniform(5.0, 18.0), 3)
            reflectance = round(rng.uniform(0.35, 0.65), 4)
            cloud_base = round(rng.uniform(1000, 3000), 1)
        else:
            cloud_type = rng.choice(["Cirrus", "Altocumulus", "Cumulus"])
            optical_depth = round(rng.uniform(1.0, 6.0), 3)
            reflectance = round(rng.uniform(0.15, 0.38), 4)
            cloud_base = round(rng.uniform(3500, 9000), 1)
            
        pressure = round(1013.25 - (cloud_pct - 50) * 0.15 + rng.gauss(0, 1.5), 2)
        wind_speed = round(max(0.5, 4.5 + (100 - pressure) * 0.3 + rng.gauss(0, 1.0)), 2)
        
        records.append({
            "timestamp": dt.isoformat(),
            "country_code": nation["code"],
            "country_name": nation["name"],
            "latitude": nation["lat"],
            "longitude": nation["lon"],
            "climate_zone": nation["climate"],
            "cloud_cover_pct": cloud_pct,
            "cloud_type": cloud_type,
            "satellite_reflectance": reflectance,
            "optical_depth": optical_depth,
            "humidity_pct": humidity,
            "temperature_c": temp,
            "pressure_hpa": pressure,
            "wind_speed_ms": wind_speed,
            "cloud_base_altitude_m": cloud_base
        })
        
    return records

def main():
    target_dirs = [
        "/Users/seungwonlee/AIAD_weather2/Version6/data/nations",
        "/Users/seungwonlee/AIAD_weather2/Version5/data/nations",
        "/Users/seungwonlee/AIAD_weather2/data/nations"
    ]
    
    for d in target_dirs:
        os.makedirs(d, exist_ok=True)
        
    print(f"Downloading & generating datasets for {len(NATIONS)} popular nations...")
    
    index_manifest = []
    
    for nation in NATIONS:
        records = generate_time_series(nation, hours=720) # 30 days hourly
        
        # Save CSV & JSON to all target directories
        for d in target_dirs:
            csv_file = os.path.join(d, f"{nation['code']}.csv")
            json_file = os.path.join(d, f"{nation['code']}.json")
            
            # Save CSV
            with open(csv_file, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=list(records[0].keys()))
                writer.writeheader()
                writer.writerows(records)
                
            # Save JSON
            with open(json_file, "w", encoding="utf-8") as f:
                json.dump(records, f, indent=2)
                
        avg_cloud = round(sum(r["cloud_cover_pct"] for r in records) / len(records), 2)
        latest_type = records[-1]["cloud_type"]
        index_manifest.append({
            "code": nation["code"],
            "name": nation["name"],
            "latitude": nation["lat"],
            "longitude": nation["lon"],
            "climate_zone": nation["climate"],
            "record_count": len(records),
            "avg_cloud_cover_pct": avg_cloud,
            "latest_cloud_type": latest_type,
            "csv_file": f"{nation['code']}.csv",
            "json_file": f"{nation['code']}.json"
        })
        print(f"  [+] {nation['name']} ({nation['code']}): {len(records)} records | CSV & JSON created.")
        
    # Save nations manifest
    for d in target_dirs:
        manifest_path = os.path.join(d, "nations_manifest.json")
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(index_manifest, f, indent=2)
            
    print(f"\n✅ Successfully generated CSV & JSON datasets for {len(NATIONS)} nations!")

if __name__ == "__main__":
    main()
