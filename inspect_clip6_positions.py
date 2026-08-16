import json

with open("human_preferences.json") as f:
    prefs = json.load(f)

for p in prefs:
    if p["ClipAId"] == 6 or p["ClipBId"] == 6:
        side = "A" if p["ClipAId"] == 6 else "B"
        winner_side = "A" if p["WinnerId"] == p["ClipAId"] else "B"
        print(f"Clip 6 was shown as {side}, you chose {winner_side} (clip {p['WinnerId']})")