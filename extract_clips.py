import gymnasium as gym
import numpy as np
import json
from stable_baselines3 import SAC

env = gym.make("Pendulum-v1")
trained_model = SAC.load("sac_pendulum")

CLIP_LENGTH = 50  # steps per clip
NUM_CLIPS_PER_SOURCE = 10

def collect_clip(use_trained_policy):
    state, info = env.reset()
    angles = []
    total_reward = 0

    for step in range(CLIP_LENGTH):
        if use_trained_policy:
            action, _ = trained_model.predict(state, deterministic=True)
        else:
            action = env.action_space.sample()

        state, reward, terminated, truncated, info = env.step(action)

        # state[0] = cos(theta), state[1] = sin(theta) -> reconstruct angle
        angle = np.arctan2(state[1], state[0])
        angles.append(float(angle))
        total_reward += reward

        if terminated or truncated:
            break

    return {"angles": angles, "total_reward": float(total_reward)}

clips = []
clip_id = 0

print("Collecting clips from TRAINED agent...")
for i in range(NUM_CLIPS_PER_SOURCE):
    clip = collect_clip(use_trained_policy=True)
    clip["clip_id"] = clip_id
    clip["source"] = "trained"
    clips.append(clip)
    clip_id += 1

print("Collecting clips from RANDOM agent...")
for i in range(NUM_CLIPS_PER_SOURCE):
    clip = collect_clip(use_trained_policy=False)
    clip["clip_id"] = clip_id
    clip["source"] = "random"
    clips.append(clip)
    clip_id += 1

with open("clips.json", "w") as f:
    json.dump(clips, f, indent=2)

print(f"\nSaved {len(clips)} clips to clips.json")
for c in clips:
    print(f"Clip {c['clip_id']} ({c['source']}): total_reward = {c['total_reward']:.1f}")

env.close()