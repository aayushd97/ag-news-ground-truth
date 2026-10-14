import pandas as pd, os
from sklearn.metrics import cohen_kappa_score, confusion_matrix
from statsmodels.stats.proportion import proportion_confint

mine = pd.read_csv("data/c7_to_label.csv")
hidden = pd.read_csv(os.path.expanduser("~/Desktop/c7_hidden.csv"))
codes = {"Entire": "Entire home/apt", "Private": "Private room", "Hotel": "Hotel room", "Shared": "Shared room"}
mine["my_label"] = mine["my_label"].str.strip().map(codes)
m = mine.merge(hidden, on="id")
assert m["my_label"].notna().all(), "Some labels are missing or misspelled"

n = len(m)
k = int((m["my_label"] == m["room_type"]).sum())
lo, hi = proportion_confint(k, n, method="wilson")
print(f"Agreement: {k}/{n} = {k/n:.3f}  (95% CI {lo:.2f} to {hi:.2f})")
print("Kappa:", round(cohen_kappa_score(m["my_label"], m["room_type"]), 3))
print("\nRows = host label, columns = my label")
print(pd.crosstab(m["room_type"], m["my_label"]))
print("\nBy my confidence:")
print(m.assign(match=m["my_label"] == m["room_type"]).groupby("my_confidence")["match"].agg(["mean", "count"]))