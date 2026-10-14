import time, random
from classify import classify, PROMPT1

random.seed(1)
vocab = ["banana", "carpet", "engine", "violin", "harbour", "glacier", "umbrella",
         "quantum", "pepper", "ladder", "orbit", "cushion", "tariff", "goalpost",
         "vaccine", "senate", "pixel", "marble", "turbine", "jacket"]
noise = " ".join(random.choice(vocab) for _ in range(3000))

tests = [
    ("empty string", ""),
    ("3000 words of noise", noise),
    ("Hindi headline", "भारत ने पहले टेस्ट मैच में ऑस्ट्रेलिया को हराया"),
    ("belongs in two categories", "Tech giant buys football club for record fee as its shares jump"),
]

for name, text in tests:
    r = classify(text, PROMPT1)
    print(f"--- {name} (input length {len(text)} characters)")
    print("status:", r["status"], "| label:", r["label"], "| confidence:", r["confidence"])
    print("raw:", str(r["raw"])[:200])
    time.sleep(10)