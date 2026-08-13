import json
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

class RewardModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 32), nn.ReLU(),
            nn.Linear(32, 32), nn.ReLU(),
            nn.Linear(32, 1)
        )
    def forward(self, cos_sin):
        return self.net(cos_sin).sum()

reward_model = RewardModel()
reward_model.load_state_dict(torch.load("reward_model.pt"))
reward_model.eval()

with open("clips.json") as f:
    clips = json.load(f)

predicted, true, colors = [], [], []
for clip in clips:
    angles = np.array(clip["angles"])
    cos_sin = torch.tensor(np.stack([np.cos(angles), np.sin(angles)], axis=1), dtype=torch.float32)
    with torch.no_grad():
        pred = reward_model(cos_sin).item()
    predicted.append(pred)
    true.append(clip["total_reward"])
    colors.append("tab:blue" if clip["source"] == "trained" else "tab:orange")

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

axes[0].scatter(true, predicted, c=colors, label=None)
axes[0].scatter([], [], c="tab:blue", label="Trained agent clips")
axes[0].scatter([], [], c="tab:orange", label="Random agent clips")
axes[0].legend()

axes[0].set_xlabel("True total reward")
axes[0].set_ylabel("Predicted return (reward model)")
r = np.corrcoef(predicted, true)[0, 1]
axes[0].set_title(f"Reward model vs true reward\n(Pearson r = {r:.3f})")

labels = ["Baseline\n(true reward)", "Learned reward\n(Bradley-Terry)"]
means = [-155.01, -163.62]
stds = [106.29, 50.99]
axes[1].bar(labels, means, yerr=stds, capsize=8, color=["tab:blue", "tab:green"])
axes[1].set_ylabel("Mean episode reward (true reward, 10 eval episodes)")
axes[1].set_title("Final agent comparison")

plt.tight_layout()
plt.savefig("results_comparison.png", dpi=150)
print("Saved results_comparison.png")