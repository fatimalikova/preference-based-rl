import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

with open("clips.json") as f:
    clips = json.load(f)
with open("preferences.json") as f:
    preferences = json.load(f)

clips_by_id = {c["clip_id"]: c for c in clips}

class RewardModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, cos_sin):
        # cos_sin shape: (steps, 2) -> per-step reward -> sum over clip
        per_step_reward = self.net(cos_sin)  # (steps, 1)
        return per_step_reward.sum()  # scalar: predicted clip return

def clip_to_tensor(clip):
    angles = np.array(clip["angles"])
    cos_sin = np.stack([np.cos(angles), np.sin(angles)], axis=1)
    return torch.tensor(cos_sin, dtype=torch.float32)

reward_model = RewardModel()
optimizer = optim.Adam(reward_model.parameters(), lr=1e-3)
loss_fn = nn.BCEWithLogitsLoss()

EPOCHS = 200

for epoch in range(EPOCHS):
    total_loss = 0.0
    optimizer.zero_grad()

    for pref in preferences:
        clip_a = clips_by_id[pref["clip_a_id"]]
        clip_b = clips_by_id[pref["clip_b_id"]]

        R_a = reward_model(clip_to_tensor(clip_a))
        R_b = reward_model(clip_to_tensor(clip_b))

        logit = R_a - R_b  # this is what BCEWithLogitsLoss expects
        label = torch.tensor(1.0 if pref["winner_id"] == pref["clip_a_id"] else 0.0)

        loss = loss_fn(logit, label)
        loss.backward()
        total_loss += loss.item()

    optimizer.step()

    if (epoch + 1) % 20 == 0:
        avg_loss = total_loss / len(preferences)
        print(f"Epoch {epoch+1}: avg Bradley-Terry loss = {avg_loss:.4f}")

# --- Validation: does predicted return correlate with TRUE reward? ---
print("\nValidation (predicted vs true reward, model never saw true reward directly):")
predicted = []
true = []
for clip in clips:
    with torch.no_grad():
        pred_return = reward_model(clip_to_tensor(clip)).item()
    predicted.append(pred_return)
    true.append(clip["total_reward"])
    print(f"Clip {clip['clip_id']} ({clip['source']}): predicted={pred_return:.2f}, true={clip['total_reward']:.2f}")

correlation = np.corrcoef(predicted, true)[0, 1]
print(f"\nPearson correlation (predicted vs true): {correlation:.3f}")

torch.save(reward_model.state_dict(), "reward_model.pt")
print("Saved reward_model.pt")