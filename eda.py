import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("feature_engineered_data.csv")

plt.figure(figsize=(8,5))
plt.hist(df["magnitude"], bins=20)
plt.title("Earthquake Magnitude Distribution")
plt.xlabel("Magnitude")
plt.ylabel("Count")
plt.show()
plt.figure(figsize=(10,6))
plt.scatter(
    df["longitude"],
    df["latitude"],
    s=5,
    alpha=0.5
)

plt.title("Earthquake Locations")
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.show()