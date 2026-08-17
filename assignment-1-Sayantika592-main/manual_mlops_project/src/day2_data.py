import pandas as pd
import numpy as np

df = pd.read_csv("data/processed/v1_cleaned.csv")

# future production data with drift (day 2) 
day2 = df.iloc[7000:].copy()

# slightly hotter environment
day2["Air temperature [K]"] *= 1.02
day2["Process temperature [K]"] *= 1.02

# torque little higher
day2["Torque [Nm]"] *= 1.05

# tools more worn
day2["Tool wear [min]"] += 10

# small random rpm fluctuations
day2["Rotational speed [rpm]"] += np.random.normal(0, 20, len(day2))

day2.to_csv("data/production/day2.csv", index=False)

print("Day 2 production data with drift created!")
