import json
import numpy as np

with open("clips.json") as f:
    clips = json.load(f)

print(f"{'Clip':<6}{'Source':<10}{'True reward':<14}{'Angle std (stillness)':<22}")
for c in clips:
    angles = np.array(c["angles"])
    print(f"{c['clip_id']:<6}{c['source']:<10}{c['total_reward']:<14.1f}{angles.std():<22.3f}")