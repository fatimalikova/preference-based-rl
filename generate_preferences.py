import json
import random

with open("clips.json") as f:
    clips = json.load(f)

NUM_PAIRS = 100
random.seed(42)  # reproducibility

preferences = []
for _ in range(NUM_PAIRS):
    a, b = random.sample(clips, 2)
    if a["total_reward"] == b["total_reward"]:
        continue  # skip rare exact ties

    winner_id = a["clip_id"] if a["total_reward"] > b["total_reward"] else b["clip_id"]
    preferences.append({
        "clip_a_id": a["clip_id"],
        "clip_b_id": b["clip_id"],
        "winner_id": winner_id
    })

with open("preferences.json", "w") as f:
    json.dump(preferences, f, indent=2)

print(f"Generated {len(preferences)} preference pairs -> preferences.json")