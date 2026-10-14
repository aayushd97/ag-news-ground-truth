import sys, time, csv, os
import pandas as pd
from classify import classify, PROMPT1, PROMPT2

which = sys.argv[1]                      # "main" or "imbalanced"
# usecols keeps the official-label columns out of memory entirely
df = pd.read_csv("week1_data.csv", usecols=["set", "id", "text"])
rows = df[df["set"] == which]
prompts = {"p1": PROMPT1, "p2": PROMPT2} if which == "main" else {"p1": PROMPT1}

os.makedirs("results", exist_ok=True)
out_path = f"results/ai_{which}.csv"
with open(out_path, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["id", "prompt", "ai_label", "ai_confidence", "status", "raw"])
    for _, row in rows.iterrows():
        for name, template in prompts.items():
            res = classify(row["text"], template)
            w.writerow([row["id"], name, res["label"], res["confidence"], res["status"], res["raw"]])
            f.flush()
            print(row["id"], name, res["status"], res["label"])
            time.sleep(4)                # free tier: 30 requests per minute

out = pd.read_csv(out_path)
print("\nParse failure rate by prompt:")
print(out.groupby("prompt")["status"].apply(lambda s: round((s != "ok").mean(), 3)))