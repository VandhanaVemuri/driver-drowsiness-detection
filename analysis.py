import pandas as pd
import matplotlib.pyplot as plt

print("\n===== ADVANCED DROWSINESS SYSTEM ANALYSIS =====\n")

# -----------------------------
# Load Data
# -----------------------------
try:
    df = pd.read_csv("drowsiness_log.csv")
except FileNotFoundError:
    print("Log file not found. Run main.py first.")
    exit()

# -----------------------------
# Convert Timestamp
# -----------------------------
df["Timestamp"] = pd.to_datetime(df["Timestamp"])
df = df.sort_values("Timestamp")

print("Total Events Detected:", len(df))
print("\nEvent Breakdown:")
print(df["Event"].value_counts())

# -----------------------------
# Dynamic Fatigue Model
# -----------------------------
fatigue_score = 0
scores = []
previous_time = df["Timestamp"].iloc[0]

for index, row in df.iterrows():
    current_time = row["Timestamp"]
    time_diff = (current_time - previous_time).total_seconds()

    # Fatigue decay (0.1 per second)
    fatigue_score -= 0.1 * time_diff
    fatigue_score = max(fatigue_score, 0)

    # Add event weight
    if row["Event"] == "Drowsiness":
        fatigue_score += 2
    elif row["Event"] == "Yawning":
        fatigue_score += 1

    scores.append(fatigue_score)
    previous_time = current_time

df["Dynamic_Fatigue_Score"] = scores

# -----------------------------
# Plot Dynamic Fatigue
# -----------------------------
plt.figure()
plt.plot(df["Timestamp"], df["Dynamic_Fatigue_Score"], marker='o')
plt.title("Dynamic Fatigue Score Over Time")
plt.xlabel("Time")
plt.ylabel("Fatigue Score")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# -----------------------------
# Statistical Summary
# -----------------------------
print("\n===== STATISTICAL SUMMARY =====")

print("Average EAR (Drowsiness events):")
print(df[df["Event"] == "Drowsiness"]["Value"].mean())

print("\nAverage MAR (Yawning events):")
print(df[df["Event"] == "Yawning"]["Value"].mean())
