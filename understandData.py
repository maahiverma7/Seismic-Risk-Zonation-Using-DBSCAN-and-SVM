import pandas as pd

df = pd.read_csv("India_Subcontinent_Seismic_Data_10_Years.csv")

print("\nDataset Shape:")
print(df.shape)

print("\nColumns:")
print(df.columns.tolist())

print("\nFirst 5 Rows:")
print(df.head())

print("\nMissing Values:")
print(df.isnull().sum())

print("\nData Types:")
print(df.dtypes)

print("\nStatistics:")
print(df.describe())