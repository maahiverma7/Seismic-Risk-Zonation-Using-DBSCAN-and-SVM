import pandas as pd

# Load data
df = pd.read_csv("India_Subcontinent_Seismic_Data_10_Years.csv")

# Convert time column
df["time"] = pd.to_datetime(df["time"])

# Remove duplicates
df = df.drop_duplicates()

# Create time features
df["year"] = df["time"].dt.year
df["month"] = df["time"].dt.month
df["day"] = df["time"].dt.day

print("Shape after preprocessing:")
print(df.shape)

# Save
df.to_csv("cleaned_seismic_data.csv", index=False)

print("Saved cleaned dataset.")