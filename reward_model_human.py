import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

with open("clips.json") as f:
    clips = json.load(f)
with open("human_preferences.json") as f:
    human_preferences = json.load(f)

clips_by_id = {c["clip_id"]: c for c in clips}

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

def clip_to_tensor(clip):
    angles = np.array(clip["angles"])
    cos_sin = np.stack([np.cos(angles), np.sin(angles)], axis=1)
    return torch.tensor(cos_sin, dtype=torch.float32)

reward_model = RewardModel()
optimizer = optim.Adam(reward_model.parameters(), lr=1e-3)
loss_fn = nn.BCEWithLogitsLoss()

EPOCHS = 200

print(f"Training on {len(human_preferences)} human preference pairs...")

for epoch in range(EPOCHS):
    total_loss = 0.0
    optimizer.zero_grad()

    for pref in human_preferences:
        clip_a = clips_by_id[pref["ClipAId"]]
        clip_b = clips_by_id[pref["ClipBId"]]

        R_a = reward_model(clip_to_tensor(clip_a))
        R_b = reward_model(clip_to_tensor(clip_b))

        logit = R_a - R_b
        label = torch.tensor(1.0 if pref["WinnerId"] == pref["ClipAId"] else 0.0)

        loss = loss_fn(logit, label)
        loss.backward()
        total_loss += loss.item()

    optimizer.step()

    if (epoch + 1) % 20 == 0:
        avg_loss = total_loss / len(human_preferences)
        print(f"Epoch {epoch+1}: avg Bradley-Terry loss = {avg_loss:.4f}")

print("\nValidation (predicted vs true reward):")
predicted, true = [], []
for clip in clips:
    with torch.no_grad():
        pred_return = reward_model(clip_to_tensor(clip)).item()
    predicted.append(pred_return)
    true.append(clip["total_reward"])
    print(f"Clip {clip['clip_id']} ({clip['source']}): predicted={pred_return:.2f}, true={clip['total_reward']:.2f}")

correlation = np.corrcoef(predicted, true)[0, 1]
print(f"\nPearson correlation (human-trained model vs true reward): {correlation:.3f}")
print(f"(For comparison, scripted-oracle model achieved: 0.978)")

torch.save(reward_model.state_dict(), "reward_model_human.pt")
print("Saved reward_model_human.pt")