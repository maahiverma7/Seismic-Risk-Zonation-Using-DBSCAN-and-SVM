import pandas as pd
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

df = pd.read_csv("feature_engineered_data.csv")
print(f"Loaded: {df.shape[0]} rows")

# ── Design decision: cluster on geography only ────────────────────────────────
# DBSCAN here finds dense geographic zones (fault lines, seismic belts).
# Seismic features (energy, depth) feed risk scoring AFTER clustering —
# this avoids mixing physical distance with dimensionless indexes in one space.
X = df[["latitude", "longitude"]].values
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ── K-Distance graph — justified eps selection ────────────────────────────────
k = 5
print(f"\nComputing k-distance graph (k={k}) to select eps...")
nbrs = NearestNeighbors(n_neighbors=k, algorithm="ball_tree").fit(X_scaled)
distances, _ = nbrs.kneighbors(X_scaled)
k_distances = np.sort(distances[:, k - 1])

# Search for elbow in the 10–70% percentile range to skip the flat part and the tail
search = k_distances[int(0.10 * len(k_distances)) : int(0.70 * len(k_distances))]
d2 = np.gradient(np.gradient(search))
elbow_local = np.argmax(d2)
elbow_global = elbow_local + int(0.10 * len(k_distances))
optimal_eps = round(float(k_distances[elbow_global]), 3)
print(f"  Elbow at index {elbow_global}  →  eps = {optimal_eps}")

# Plot k-distance graph
plt.figure(figsize=(10, 4))
plt.plot(k_distances, color="#E74C3C", linewidth=1.5, label="5th neighbour distance")
plt.axvline(x=elbow_global, color="gray", linestyle="--", linewidth=1, alpha=0.6)
plt.axhline(y=optimal_eps, color="#3498DB", linestyle="--", linewidth=1.2,
            label=f"Selected eps = {optimal_eps}")
plt.title(f"K-Distance Graph — Optimal eps Selection (k={k})")
plt.xlabel("Points sorted by k-th nearest distance")
plt.ylabel("Distance to 5th Nearest Neighbour")
plt.legend()
plt.tight_layout()
plt.savefig("kdistance_graph.png", dpi=150)
plt.close()
print("  Saved → kdistance_graph.png")

# ── DBSCAN ───────────────────────────────────────────────────────────────────
# eps=0.05 (standardised lat/lon) ≈ ~30-40 km radius; min_samples=8 per cluster
eps_to_use   = 0.05
min_samples  = 8

print(f"\nRunning DBSCAN  eps={eps_to_use}  min_samples={min_samples}")
dbscan = DBSCAN(eps=eps_to_use, min_samples=min_samples, n_jobs=-1)
clusters = dbscan.fit_predict(X_scaled)

df["cluster"] = clusters
n_clusters = len(set(clusters)) - (1 if -1 in clusters else 0)
n_noise    = int((clusters == -1).sum())
noise_pct  = round(100 * n_noise / len(clusters), 1)

print(f"\n  Clusters found : {n_clusters}")
print(f"  Noise points   : {n_noise}  ({noise_pct}%)")
print(f"  Clustered pts  : {len(clusters) - n_noise}")
print(f"\nTop 10 clusters by size:")
print(df["cluster"].value_counts().head(10).to_string())

# ── Scatter plot ──────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(12, 8))
noise_mask = df["cluster"] == -1
ax.scatter(df.loc[noise_mask, "longitude"], df.loc[noise_mask, "latitude"],
           c="lightgray", s=4, alpha=0.5, label=f"Noise ({n_noise} pts)")
non_noise = df[~noise_mask]
scatter = ax.scatter(non_noise["longitude"], non_noise["latitude"],
                     c=non_noise["cluster"], cmap="tab20", s=12, alpha=0.75)
plt.colorbar(scatter, ax=ax, label="Cluster ID")
ax.set_title(f"DBSCAN Seismic Hotspot Clusters  (n={n_clusters}, eps={eps_to_use})")
ax.set_xlabel("Longitude");  ax.set_ylabel("Latitude")
ax.legend(markerscale=2, loc="lower left")
plt.tight_layout()
plt.savefig("dbscan_clusters.png", dpi=150)
plt.close()
print("  Saved → dbscan_clusters.png")

df.to_csv("clustered_data.csv", index=False)
print(f"\nSaved → clustered_data.csv")
