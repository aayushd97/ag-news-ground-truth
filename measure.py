import pandas as pd
from sklearn.metrics import cohen_kappa_score, confusion_matrix, classification_report, accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from statsmodels.stats.proportion import proportion_confint

LABELS = ["World", "Sports", "Business", "Sci/Tech"]
pd.set_option("display.width", 200)
pd.set_option("display.max_colwidth", 110)

df = pd.read_csv("week1_data.csv")
main = df[df["set"] == "main"][["id", "text", "official_label"]]
mine = pd.read_csv("data/my_labels.csv")
ai = pd.read_csv("results/ai_main.csv")

lab = ai.pivot(index="id", columns="prompt", values="ai_label").reset_index()
lab.columns = ["id", "p1_label", "p2_label"]
conf = ai.pivot(index="id", columns="prompt", values="ai_confidence").reset_index()
conf.columns = ["id", "p1_conf", "p2_conf"]

m = main.merge(mine, on="id").merge(lab, on="id").merge(conf, on="id")
for c in ["p1_label", "p2_label"]:
    m[c] = m[c].fillna("FAILED")          # failed replies count as disagreements
m.to_csv("results/main_merged.csv", index=False)
n = len(m)

print("=" * 60, "\n1. AGREEMENT AND KAPPA (n = %d)" % n)
pairs = [("my_label", "official_label"), ("p1_label", "official_label"),
         ("p2_label", "official_label"), ("my_label", "p1_label"), ("my_label", "p2_label")]
rows = []
for a, b in pairs:
    agree = (m[a] == m[b]).mean()
    k = cohen_kappa_score(m[a], m[b])
    lo, hi = proportion_confint(int((m[a] == m[b]).sum()), n, method="wilson")
    rows.append([f"{a} vs {b}", round(agree, 3), round(k, 3), f"{lo:.2f} to {hi:.2f}"])
print(pd.DataFrame(rows, columns=["pair", "agreement", "kappa", "95% Wilson CI"]).to_string(index=False))

print("\n" + "=" * 60, "\n2. CONFUSION MATRICES (rows = official, columns = predicted)")
for col, name in [("my_label", "ME"), ("p1_label", "AI PROMPT 1")]:
    cm = pd.DataFrame(confusion_matrix(m["official_label"], m[col], labels=LABELS),
                      index=LABELS, columns=LABELS)
    print(f"\n{name} vs official\n{cm}")
    off = [(cm.index[i], cm.columns[j], cm.iloc[i, j])
           for i in range(4) for j in range(4) if i != j and cm.iloc[i, j] > 0]
    off.sort(key=lambda t: -t[2])
    print("Most confused (official -> predicted):", off[:3])
    print(classification_report(m["official_label"], m[col], labels=LABELS, zero_division=0))

print("=" * 60, "\n3. ROWS WHERE I AND PROMPT 1 AGREE BUT OFFICIAL DIFFERS (write-up question 3)")
q3 = m[(m["my_label"] == m["p1_label"]) & (m["my_label"] != m["official_label"])]
print(q3[["id", "my_label", "p1_label", "official_label", "text"]].to_string(index=False))

print("\n" + "=" * 60, "\n4. CALIBRATION")
m["me_correct"] = m["my_label"] == m["official_label"]
print("My confidence vs matched official:")
print(m.groupby("my_confidence")["me_correct"].agg(["mean", "count"]))
m["p1_correct"] = m["p1_label"] == m["official_label"]
m["p1_bin"] = pd.cut(m["p1_conf"], [0, 0.8, 0.95, 1.0])
print("\nAI prompt 1 confidence vs matched official:")
print(m.groupby("p1_bin", observed=True)["p1_correct"].agg(["mean", "count"]))

print("\n" + "=" * 60, "\n5. IMBALANCED SET")
imb = df[df["set"] == "imbalanced"][["id", "official_label"]].merge(
    pd.read_csv("results/ai_imbalanced.csv")[["id", "ai_label"]], on="id")
imb["ai_label"] = imb["ai_label"].fillna("FAILED")
print("Class counts:", imb["official_label"].value_counts().to_dict())
print("AI accuracy:", round(accuracy_score(imb["official_label"], imb["ai_label"]), 3),
      "| kappa:", round(cohen_kappa_score(imb["official_label"], imb["ai_label"]), 3))
always = ["World"] * len(imb)
print("Always-World accuracy:", round(accuracy_score(imb["official_label"], always), 3),
      "| kappa:", round(cohen_kappa_score(imb["official_label"], always), 3))
print(classification_report(imb["official_label"], imb["ai_label"], labels=LABELS, zero_division=0))

print("=" * 60, "\n6. LEAKAGE WITH TF-IDF + LOGISTIC REGRESSION")
full = df[df["set"] == "full"][["text", "official_label"]]
held = set(main["text"]) | set(df[df["set"] == "imbalanced"]["text"])
print("Rows in 'full' that also appear in main or imbalanced:", int(full["text"].isin(held).sum()))

def fit_score(train, test):
    vec = TfidfVectorizer(sublinear_tf=True, ngram_range=(1, 2), min_df=2)
    clf = LogisticRegression(max_iter=1000).fit(vec.fit_transform(train["text"]), train["official_label"])
    return accuracy_score(test["official_label"], clf.predict(vec.transform(test["text"])))

test30 = main[["text", "official_label"]]
leaky = pd.concat([full, test30])
honest = full[~full["text"].isin(held)]
print("Demo A, scored on your 30 main items")
print("  leaky  (30 items inside training data):", round(fit_score(leaky, test30), 3))
print("  honest (those items removed):", round(fit_score(honest, test30), 3))

train, hold = train_test_split(full, train_size=6000, random_state=42)
print("Demo B, 6000 train rows vs 1600 unseen rows")
print("  accuracy on the 6000 it trained on:", round(fit_score(train, train), 3))
print("  accuracy on the 1600 it never saw :", round(fit_score(train, hold), 3))

