import pandas as pd
import folium
from folium.plugins import HeatMap, MarkerCluster

df = pd.read_csv("risk_labeled_data.csv")
print(f"Loaded: {df.shape[0]} rows")

risk_colors = {"High": "red", "Medium": "orange", "Low": "green"}

# ── Map 1: Full risk map with legend ─────────────────────────────────────────
m = folium.Map(location=[28.0, 82.0], zoom_start=5, tiles="CartoDB positron")

# Legend HTML
legend_html = """
<div style="position:fixed;bottom:40px;left:40px;z-index:1000;background:white;
     padding:12px 16px;border-radius:8px;border:1px solid #ccc;font-family:sans-serif;font-size:13px;">
  <b>Seismic Risk Level</b><br>
  <span style="color:red;">&#9679;</span> High<br>
  <span style="color:orange;">&#9679;</span> Medium<br>
  <span style="color:green;">&#9679;</span> Low<br>
  <hr style="margin:6px 0;">
  <span style="color:#555;font-size:11px;">DBSCAN + SVM | 10yr USGS data</span>
</div>
"""
m.get_root().html.add_child(folium.Element(legend_html))

# Plot ALL points (no sampling)
for _, row in df.iterrows():
    color = risk_colors.get(row["risk_label"], "gray")
    folium.CircleMarker(
        location=[row["latitude"], row["longitude"]],
        radius=4,
        color=color,
        fill=True,
        fill_color=color,
        fill_opacity=0.7,
        popup=folium.Popup(
            f"<b>Risk:</b> {row['risk_label']}<br>"
            f"<b>Magnitude:</b> {row['magnitude']}<br>"
            f"<b>Depth:</b> {row['depth_km']:.1f} km<br>"
            f"<b>Place:</b> {row['place']}<br>"
            f"<b>Cluster:</b> {int(row['cluster'])}<br>"
            f"<b>Risk Score:</b> {row['risk_score']:.1f}",
            max_width=220
        )
    ).add_to(m)

m.save("seismic_risk_map.html")
print("Saved → seismic_risk_map.html  (all points, with legend)")

# ── Map 2: Heatmap of seismic activity ────────────────────────────────────────
m2 = folium.Map(location=[28.0, 82.0], zoom_start=5, tiles="CartoDB dark_matter")
heat_data = [[row["latitude"], row["longitude"], row["magnitude"]]
             for _, row in df.iterrows()]
HeatMap(heat_data, min_opacity=0.3, max_zoom=10, radius=12, blur=8).add_to(m2)
m2.save("seismic_heatmap.html")
print("Saved → seismic_heatmap.html   (magnitude-weighted heatmap)")

print("\nDone.")
