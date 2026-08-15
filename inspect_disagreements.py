import json

with open("clips.json") as f:
    clips = json.load(f)
with open("human_preferences.json") as f:
    prefs = json.load(f)

clips_by_id = {c["clip_id"]: c for c in clips}

print("Disagreements (you chose X, but X had LOWER true reward):\n")
for p in prefs:
    a = clips_by_id[p["ClipAId"]]
    b = clips_by_id[p["ClipBId"]]
    true_winner = a["clip_id"] if a["total_reward"] > b["total_reward"] else b["clip_id"]
    if p["WinnerId"] != true_winner:
        chosen = clips_by_id[p["WinnerId"]]
        other_id = a["clip_id"] if b["clip_id"] == p["WinnerId"] else b["clip_id"]
        other = clips_by_id[other_id]
        print(f"You picked clip {chosen['clip_id']} (true_reward={chosen['total_reward']:.1f}, source={chosen['source']}) "
              f"over clip {other['clip_id']} (true_reward={other['total_reward']:.1f}, source={other['source']})")