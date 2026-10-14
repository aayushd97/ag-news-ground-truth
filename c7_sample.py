import pandas as pd, os

SRC = os.path.expanduser("~/Desktop/test.csv")
df = pd.read_csv(SRC, usecols=["id", "city", "room_type", "description"])

# English-language cities only, so you can read every description
pool = df[df["city"].isin(["London", "Los Angeles", "New York City", "Hawaii"])
          & (df["description"].str.len() >= 100)]

plan = {"Entire home/apt": 10, "Private room": 10, "Hotel room": 5, "Shared room": 5}
parts = [pool[pool["room_type"] == k].sample(n, random_state=42) for k, n in plan.items()]
s = pd.concat(parts).sample(frac=1, random_state=42).reset_index(drop=True)   # shuffle

# File you label: no room_type column
out = s[["id", "description"]].copy()
out["my_label"] = ""
out["my_confidence"] = ""
os.makedirs("data", exist_ok=True)
out.to_csv("data/c7_to_label.csv", index=False)

# Hidden answers go to the Desktop, outside the repo. Do not open this file yet.
s[["id", "room_type"]].to_csv(os.path.expanduser("~/Desktop/c7_hidden.csv"), index=False)
print("Done. Label data/c7_to_label.csv. Do not open ~/Desktop/c7_hidden.csv.")