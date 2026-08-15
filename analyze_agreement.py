import json

with open("clips.json") as f:
    clips = json.load(f)
with open("human_preferences.json") as f:
    prefs = json.load(f)

clips_by_id = {c["clip_id"]: c for c in clips}

agree = 0
total = 0

for p in prefs:
    a = clips_by_id[p["ClipAId"]]
    b = clips_by_id[p["ClipBId"]]
    true_winner = a["clip_id"] if a["total_reward"] > b["total_reward"] else b["clip_id"]
    if p["WinnerId"] == true_winner:
        agree += 1
    total += 1

print(f"Human choice matched true-reward ranking in {agree}/{total} pairs ({100*agree/total:.1f}%)")