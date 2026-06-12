import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import BallTree
import os

def run_feature_engineering():
    print("[INFO] Initializing Scientific Feature Engineering Pipeline...")
    src_path = "cleaned_seismic_data.csv"
    if not os.path.exists(src_path):
        raise FileNotFoundError(f"Missing base data: {src_path}. Run preprocessData.py first.")
        
    df = pd.read_csv(src_path)
    df["time"] = pd.to_datetime(df["time"])
    
    # 1. Gutenberg-Richter Energy Release Calculation
    # log10(E) = 5.24 + 1.44 * M
    log_E = 5.24 + 1.44 * df["magnitude"]
    df["energy_joules"] = 10 ** log_E
    
    scaler = MinMaxScaler()
    df["energy_index"] = scaler.fit_transform(log_E.values.reshape(-1, 1)).flatten()
    
    # 2. Depth Danger Score (Inverted Exponential Decay)
    depth_inverted = 1.0 / (df["depth_km"] + 1.0)
    df["depth_score"] = scaler.fit_transform(depth_inverted.values.reshape(-1, 1)).flatten()
    
    # 3. Seismic Frequency Index via BallTree Haversine Metrics
    # Convert degrees to radians for proper spherical distance calculations
    coords_rad = np.radians(df[["latitude", "longitude"]].values)
    tree = BallTree(coords_rad, metric="haversine")
    
    # Radius: 111 km approximated to Earth radius (6371 km) => ~0.01742 radians
    earth_radius_km = 6371.0
    target_radius_km = 111.0
    radius_radians = target_radius_km / earth_radius_km
    
    counts = tree.query_radius(coords_rad, r=radius_radians, count_only=True)
    df["frequency_index"] = scaler.fit_transform(counts.reshape(-1, 1)).flatten()
    
    # 4. Magnitude Unpredictability / Variability
    # Computed using a rolling window over historical event sequences
    rolling_std = df["magnitude"].rolling(window=50, min_periods=1).std().fillna(0)
    df["mag_variability"] = scaler.fit_transform(rolling_std.values.reshape(-1, 1)).flatten()
    
    # 5. Composite Seismic Risk Index Construction
    # Seismologically informed weight distributions
    w_energy = 0.35
    w_depth = 0.25
    w_frequency = 0.25
    w_variability = 0.15
    
    df["composite_score"] = (
        w_energy * df["energy_index"] +
        w_depth * df["depth_score"] +
        w_frequency * df["frequency_index"] +
        w_variability * df["mag_variability"]
    )
    
    out_path = "feature_engineered_data.csv"
    df.to_csv(out_path, index=False)
    print(f"[SUCCESS] Feature Engineering completed. Target file saved: {out_path} with shape {df.shape}")

if __name__ == "__main__":
    run_feature_engineering()