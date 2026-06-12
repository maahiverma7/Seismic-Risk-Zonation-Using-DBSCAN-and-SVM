import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

df = pd.read_csv("clustered_data.csv")
print(f"Loaded: {df.shape[0]} rows, {df['cluster'].nunique()-1} clusters")

# ── Per-cluster risk scoring ──────────────────────────────────────────────────
clustered = df[df["cluster"] != -1].copy()

cluster_stats = clustered.groupby("cluster").agg(
    avg_magnitude   = ("magnitude",       "mean"),
    max_magnitude   = ("magnitude",       "max"),
    cluster_size    = ("magnitude",       "count"),
    avg_depth       = ("depth_km",        "mean"),
    avg_energy      = ("energy_index",    "mean"),
    avg_freq        = ("frequency_index", "mean"),
).reset_index()

# Depth: shallower = higher surface danger (inverted)
cluster_stats["depth_risk"] = 1.0 / (cluster_stats["avg_depth"] + 1.0)

# Normalize all metrics to [0, 1]
scaler = MinMaxScaler()
norm_cols = ["avg_magnitude", "max_magnitude", "cluster_size",
             "avg_energy", "avg_freq", "depth_risk"]
for col in norm_cols:
    cluster_stats[f"{col}_norm"] = scaler.fit_transform(
        cluster_stats[[col]]
    ).flatten()

# Composite risk score (weights grounded in seismology literature)
cluster_stats["risk_score"] = (
    0.30 * cluster_stats["avg_magnitude_norm"]  +  # avg seismic strength
    0.20 * cluster_stats["avg_energy_norm"]     +  # energy release (G-R based)
    0.20 * cluster_stats["cluster_size_norm"]   +  # recurrence frequency
    0.15 * cluster_stats["depth_risk_norm"]     +  # shallow = more destructive
    0.10 * cluster_stats["avg_freq_norm"]       +  # local density
    0.05 * cluster_stats["max_magnitude_norm"]     # peak event severity
) * 100

# 4-tier labeling (matches India's official Zone II–V system)
def assign_risk(score):
    if score >= 70:   return "Very High"
    elif score >= 50: return "High"
    elif score >= 30: return "Medium"
    else:             return "Low"

cluster_stats["risk_label"] = cluster_stats["risk_score"].apply(assign_risk)

print("\nCluster risk distribution:")
print(cluster_stats["risk_label"].value_counts().to_string())
print(f"\nRisk score range: {cluster_stats['risk_score'].min():.1f} – {cluster_stats['risk_score'].max():.1f}")

# ── Map risk back to individual events ───────────────────────────────────────
label_map = dict(zip(cluster_stats["cluster"], cluster_stats["risk_label"]))
score_map = dict(zip(cluster_stats["cluster"], cluster_stats["risk_score"]))

df["risk_label"] = df["cluster"].map(label_map)
df["risk_score"] = df["cluster"].map(score_map)

# ── Noise points: rule-based labeling (NOT auto "Low") ───────────────────────
# Noise = DBSCAN outliers, but a M7 outlier is still dangerous.
# Apply magnitude + depth thresholds directly.
noise_mask = df["cluster"] == -1

def noise_risk(row):
    if row["magnitude"] >= 6.0:                      return "Very High"
    if row["magnitude"] >= 5.5 or row["depth_km"] < 15: return "High"
    if row["magnitude"] >= 5.0:                      return "Medium"
    return "Low"

df.loc[noise_mask, "risk_label"] = df[noise_mask].apply(noise_risk, axis=1)
df.loc[noise_mask, "risk_score"] = df[noise_mask].apply(
    lambda r: {"Very High": 80, "High": 55, "Medium": 35, "Low": 10}[noise_risk(r)],
    axis=1
)

print("\nFinal risk distribution (including noise points):")
print(df["risk_label"].value_counts().to_string())

# Verify the M7.8 event is correctly labeled
m78 = df[df["magnitude"] >= 7.0][["place","magnitude","depth_km","cluster","risk_label"]]
if not m78.empty:
    print(f"\nM7.0+ events (sanity check):")
    print(m78.to_string(index=False))

df.to_csv("risk_labeled_data.csv", index=False)
cluster_stats.to_csv("cluster_risk_stats.csv", index=False)
print("\nSaved → risk_labeled_data.csv")
print("Saved → cluster_risk_stats.csv")
