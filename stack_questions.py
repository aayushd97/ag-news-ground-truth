import pandas as pd, duckdb

pd.set_option("display.width", 200)
pd.set_option("display.max_colwidth", 100)
df = pd.read_csv("week1_data.csv", usecols=["set", "id", "text", "official_label"])
main = df[df["set"] == "main"]

print("--- 1. PANDAS: items containing 'world' ---")
hits = main[main["text"].str.contains("world", case=False)]
print(hits[["id", "official_label", "text"]].to_string(index=False))

print("\n--- 2. SQL: the same filter in DuckDB ---")
print(duckdb.query("SELECT id, official_label, text FROM main WHERE lower(text) LIKE '%world%'").df().to_string(index=False))

print("\n--- 2b. The group-by from the brief ---")
print(duckdb.query("SELECT official_label, COUNT(*) AS n, ROUND(AVG(LENGTH(text)),1) AS avg_chars FROM main GROUP BY official_label ORDER BY n DESC").df())