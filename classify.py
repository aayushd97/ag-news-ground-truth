import os, json, time, requests
from dotenv import load_dotenv
load_dotenv()

URL = "https://api.groq.com/openai/v1/chat/completions"
MODEL = "openai/gpt-oss-20b"   
LABELS = {"World", "Sports", "Business", "Sci/Tech"}

PROMPT1 = """You are classifying news items into exactly one of four categories.
World: international news, politics, conflict, disasters, national affairs
Sports: matches, results, athletes, teams, transfers, tournaments
Business: companies, markets, earnings, the economy, trade, jobs, deals
Sci/Tech: science, research, technology, software, space, health research
Boundary rules, apply them in this order:
1. A foreign company's earnings or a currency move is Business, not World.
2. A broadcasting rights deal or a club's finances is Business, not Sports.
3. A product's technical capability is Sci/Tech. Its commercial impact is Business.
4. If two categories still fit, choose the one the item is mostly about.
Reply with only JSON in this exact shape: {"label": "<one of World, Sports, Business, Sci/Tech>", "confidence": <number from 0 to 1>}
No other text.
News item:
{text}"""

PROMPT2 = """What category is this news item? Pick one.
World, Sports, Business, Sci/Tech
Reply with only JSON in this exact shape: {"label": "<category>", "confidence": <number from 0 to 1>}
News item:
{text}"""


def call_model(prompt, retries=3):
    headers = {"Authorization": f"Bearer {os.getenv('GROQ_API_KEY')}"}
    body = {"model": MODEL, "temperature": 0, "max_tokens": 400,
            "messages": [{"role": "user", "content": prompt}]}
    if "gpt-oss" in MODEL:
        body["reasoning_effort"] = "low"   # keeps the reasoning from eating the token budget
    r = None
    for _ in range(retries):
        r = requests.post(URL, headers=headers, json=body, timeout=30)
        if r.status_code == 429:
            time.sleep(float(r.headers.get("retry-after", 10)))
            continue
        return r
    return r


def classify(text, template):
    r = call_model(template.replace("{text}", text))
    if r.status_code != 200:
        return {"label": None, "confidence": None, "status": f"http_{r.status_code}", "raw": r.text[:200]}
    raw = r.json()["choices"][0]["message"]["content"] or ""
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return {"label": None, "confidence": None, "status": "parse_error", "raw": raw}
    label = data.get("label") if isinstance(data, dict) else None
    conf = data.get("confidence") if isinstance(data, dict) else None
    if label not in LABELS:
        return {"label": None, "confidence": None, "status": "bad_label", "raw": raw}
    if isinstance(conf, bool) or not isinstance(conf, (int, float)) or not 0 <= conf <= 1:
        return {"label": None, "confidence": None, "status": "bad_confidence", "raw": raw}
    return {"label": label, "confidence": conf, "status": "ok", "raw": raw}