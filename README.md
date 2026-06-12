# Seismic Risk Zonation for Disaster Preparedness

ML pipeline combining DBSCAN clustering and SVM classification to identify seismic hotspots and classify risk zones across the Indian subcontinent.

---

## Dataset

| Property | Value |
|---|---|
| Source | USGS Earthquake Catalog API |
| Region | India + bordering fault zones (Nepal, Afghanistan, Myanmar, N. Indonesia) |
| Period | 2014 – 2024 (10 years) |
| Filter | Magnitude ≥ 3.5 |
| Records | 5,933 earthquake events |

Bounding box includes neighboring countries deliberately — fault lines in Afghanistan's Hindu Kush and the Nepal Himalayan belt directly affect North India, and northern Indonesia covers the Andaman fault line.

---

## Pipeline

```
extractData.py          → Raw CSV from USGS API
preprocessData.py       → Deduplication, time parsing, date features
featureEngineering.py   → 5 physics-grounded features
dbscan_clustering.py    → K-distance graph + DBSCAN
riskLabelGeneration.py  → Weighted composite risk score → Low/Medium/High
svm_classification.py   → 3-kernel SVM comparison, 5-fold CV
risk_map.py             → Interactive Folium maps
```

---

## Feature Engineering

All features are computed at event level from raw properties — no group-level aggregation that would cause leakage into the test set.

| Feature | Formula / Method | Rationale |
|---|---|---|
| `energy_index` | log₁₀(E) = 5.24 + 1.44×M (Gutenberg-Richter), normalized | Physics-grounded energy release |
| `depth_score` | 1 / (depth + 1), normalized | Shallow quakes cause more surface damage |
| `frequency_index` | BallTree haversine count within 111 km radius, normalized | Local fault-line activity density |
| `mag_variability` | Rolling std (window=50), normalized | Regional seismic unpredictability |
| `composite_score` | Weighted blend (energy 35%, depth 25%, frequency 25%, variability 15%) | Pre-SVM risk signal |

---

## DBSCAN Clustering

- **Features used:** latitude, longitude only — geographic distance, not a mixed feature space
- **Eps selection:** K-distance graph (k=5) with elbow detection; validated eps=0.05 (≈30 km radius in standardised space)
- **Results:** 46 clusters, 453 noise points (7.6%)

Noise points are NOT auto-labeled "Low" — a magnitude-threshold rule assigns their risk independently.

---

## Risk Label Generation

Cluster-level composite score (0–100):

| Weight | Metric |
|---|---|
| 30% | Average magnitude |
| 20% | Energy index (G-R based) |
| 20% | Cluster size (recurrence) |
| 15% | Depth risk (inverted depth) |
| 10% | Frequency index |
| 5% | Max magnitude |

**Noise point fallback:** M ≥ 6.0 → High, M ≥ 5.5 or depth < 15 km → High, M ≥ 5.0 → Medium, else Low.

---

## SVM Classification

Three kernels trained and compared:

| Kernel | Test Accuracy | 5-Fold CV |
|---|---|---|
| Linear SVM | 75.65% | 74.15% ± 1.26% |
| Polynomial SVM (deg=3) | 78.26% | 74.86% ± 0.73% |
| **RBF SVM** | **78.01%** | **75.62% ± 1.43%** |

Best model selected by cross-validation mean: **RBF SVM** with `class_weight="balanced"`.

### Why RBF?
Seismic risk is non-linear — magnitude and depth interact in ways no straight hyperplane can capture. RBF handles non-linear boundaries while remaining robust with `gamma="scale"`.

---

## Results

| Metric | Value |
|---|---|
| Clusters found | 46 |
| Noise points | 453 (7.6%) |
| High risk events | 4,098 |
| Medium risk events | 1,572 |
| Low risk events | 263 |
| SVM Accuracy | 78.01% |
| SVM CV Score | 75.62% ± 1.43% |

---

## Output Files

| File | Description |
|---|---|
| `seismic_risk_map.html` | All 5,933 points coloured by risk level, with legend and popups |
| `seismic_heatmap.html` | Magnitude-weighted density heatmap |
| `kdistance_graph.png` | Eps selection plot |
| `dbscan_clusters.png` | Cluster scatter plot |
| `confusion_matrix.png` | SVM confusion matrix |
| `kernel_comparison.png` | Bar chart comparing all three kernels |
| `cluster_risk_stats.csv` | Per-cluster risk scores and labels |

---

## Technologies

Python · Pandas · NumPy · scikit-learn · Folium · Matplotlib · Seaborn
