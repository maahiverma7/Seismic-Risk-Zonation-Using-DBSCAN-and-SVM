import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

df = pd.read_csv("risk_labeled_data.csv")

# Drop noise-only label check
print("Risk label distribution:")
print(df["risk_label"].value_counts().to_string())
print(f"\nDataset: {df.shape[0]} rows")

# ── Features ──────────────────────────────────────────────────────────────────
# No leakage: eq_frequency and activity_score are removed.
# All features are event-level properties, not derived from the full dataset.
features = [
    "latitude", "longitude",
    "magnitude", "depth_km",
    "energy_index",      # Gutenberg-Richter based
    "depth_score",       # 1/(depth+1) normalized
    "frequency_index",   # local spatial density (BallTree)
    "mag_variability",   # rolling std of magnitude
    "composite_score"    # weighted blend of above
]

X = df[features]
y = df["risk_label"]

le = LabelEncoder()
y_enc = le.fit_transform(y)

print(f"\nClasses: {list(le.classes_)}")

# ── Train / test split ────────────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y_enc, test_size=0.2, random_state=42, stratify=y_enc
)

# Scale — fit ONLY on training data (no leakage)
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s  = scaler.transform(X_test)

# ── Compare 3 kernels ─────────────────────────────────────────────────────────
print("\nTraining and comparing three SVM kernels...\n")
kernels = {
    "Linear SVM":     SVC(kernel="linear",  C=1.0, class_weight="balanced", random_state=42),
    "Polynomial SVM": SVC(kernel="poly",    C=1.0, degree=3, class_weight="balanced", random_state=42),
    "RBF SVM":        SVC(kernel="rbf",     C=1.0, gamma="scale", class_weight="balanced", random_state=42),
}

results = {}
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

for name, model in kernels.items():
    model.fit(X_train_s, y_train)
    y_pred    = model.predict(X_test_s)
    acc       = accuracy_score(y_test, y_pred)
    cv_scores = cross_val_score(model, X_train_s, y_train, cv=cv, scoring="accuracy")
    results[name] = {
        "model": model, "accuracy": acc,
        "cv_mean": cv_scores.mean(), "cv_std": cv_scores.std(),
        "y_pred": y_pred
    }
    print(f"  {name:<20} Test acc: {acc:.4f}   CV: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

# ── Pick best kernel ──────────────────────────────────────────────────────────
best_name = max(results, key=lambda k: results[k]["cv_mean"])
best      = results[best_name]
y_pred    = best["y_pred"]
print(f"\nBest kernel: {best_name}  (CV mean = {best['cv_mean']:.4f})")

print(f"\nAccuracy  : {best['accuracy']:.4f}")
print(f"CV Score  : {best['cv_mean']:.4f} ± {best['cv_std']:.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=le.classes_))

# ── Confusion matrix plot ─────────────────────────────────────────────────────
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(7, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=le.classes_, yticklabels=le.classes_)
plt.title(f"Confusion Matrix — {best_name}")
plt.xlabel("Predicted"); plt.ylabel("Actual")
plt.tight_layout()
plt.savefig("confusion_matrix.png", dpi=150)
plt.close()
print("Saved → confusion_matrix.png")

# ── Kernel comparison bar chart ───────────────────────────────────────────────
names = list(results.keys())
accs  = [results[n]["accuracy"]  for n in names]
cvs   = [results[n]["cv_mean"]   for n in names]
x     = np.arange(len(names))
width = 0.35

fig, ax = plt.subplots(figsize=(9, 5))
bars1 = ax.bar(x - width/2, accs, width, label="Test accuracy", color="#3498DB", alpha=0.85)
bars2 = ax.bar(x + width/2, cvs,  width, label="CV mean (5-fold)", color="#2ECC71", alpha=0.85)
ax.set_xticks(x); ax.set_xticklabels(names)
ax.set_ylim(0, 1.1)
ax.set_ylabel("Accuracy")
ax.set_title("SVM Kernel Comparison — Test Accuracy vs 5-Fold CV")
ax.legend()
ax.axhline(y=max(accs), color="gray", linestyle="--", linewidth=0.8, alpha=0.5)
for bar in bars1: ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                           f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=9)
for bar in bars2: ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                           f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=9)
plt.tight_layout()
plt.savefig("kernel_comparison.png", dpi=150)
plt.close()
print("Saved → kernel_comparison.png")

# ── Save best model ───────────────────────────────────────────────────────────
joblib.dump(best["model"], "svm_model.pkl")
joblib.dump(scaler,        "scaler.pkl")
joblib.dump(le,            "label_encoder.pkl")
print(f"\nBest model ({best_name}) saved → svm_model.pkl")
