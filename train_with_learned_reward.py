import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
from stable_baselines3 import SAC
from stable_baselines3.common.evaluation import evaluate_policy

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
        return self.net(cos_sin).sum()

reward_model = RewardModel()
reward_model.load_state_dict(torch.load("reward_model.pt"))
reward_model.eval()

class LearnedRewardWrapper(gym.Wrapper):
    """Replaces the environment's true reward with the learned reward model's prediction."""
    def __init__(self, env, reward_model):
        super().__init__(env)
        self.reward_model = reward_model

    def step(self, action):
        state, true_reward, terminated, truncated, info = self.env.step(action)
        cos_theta, sin_theta = state[0], state[1]
        with torch.no_grad():
            cos_sin = torch.tensor([[cos_theta, sin_theta]], dtype=torch.float32)
            learned_reward = self.reward_model(cos_sin).item()
        info["true_reward"] = true_reward  # keep the true reward around for inspection
        return state, learned_reward, terminated, truncated, info

# Train a NEW agent using ONLY the learned reward (never sees the true reward)
env = gym.make("Pendulum-v1")
wrapped_env = LearnedRewardWrapper(env, reward_model)

model = SAC("MlpPolicy", wrapped_env, verbose=1)
model.learn(total_timesteps=20000)
model.save("sac_pendulum_learned_reward")

# Evaluate the SAME agent on the environment's TRUE reward, for a fair comparison
eval_env = gym.make("Pendulum-v1")
mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=10)

print("\n=== FINAL COMPARISON ===")
print(f"Baseline agent (trained on TRUE reward):    -155.01 +/- 106.29")
print(f"New agent (trained on LEARNED reward):       {mean_reward:.2f} +/- {std_reward:.2f}")