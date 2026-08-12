import gymnasium as gym
from stable_baselines3 import SAC
from stable_baselines3.common.evaluation import evaluate_policy

# Create the Pendulum environment
env = gym.make("Pendulum-v1")

# Create the SAC model (uses a neural network policy)
model = SAC("MlpPolicy", env, verbose=1)

# Train the model
model.learn(total_timesteps=20000)

# Evaluate performance: run 10 episodes and average the total reward
mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10)
print(f"\nMean reward over 10 episodes: {mean_reward:.2f} +/- {std_reward:.2f}")

# Save the trained model for later use
model.save("sac_pendulum")

env.close()