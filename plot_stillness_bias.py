import json
import numpy as np
import matplotlib.pyplot as plt

with open("clips.json") as f:
    clips = json.load(f)

stds = [np.array(c["angles"]).std() for c in clips]
rewards = [c["total_reward"] for c in clips]
colors = ["tab:red" if c["clip_id"] == 6 else ("tab:blue" if c["source"] == "trained" else "tab:orange") for c in clips]

plt.figure(figsize=(7, 5))
plt.scatter(stds, rewards, c=colors, s=80)
plt.annotate("Clip 6\n(rejected in all 4\ncomparisons shown)", (0.094, -2.0), xytext=(0.5, -60),
             arrowprops=dict(arrowstyle="->", color="tab:red"), color="tab:red", fontsize=9)
plt.xlabel("Angle std within clip (stillness)")
plt.ylabel("True total reward")
plt.title("Best true-reward clip was the most 'still' — and was rejected by the human labeler")
plt.tight_layout()
plt.savefig("stillness_bias.png", dpi=150)
print("Saved stillness_bias.png")