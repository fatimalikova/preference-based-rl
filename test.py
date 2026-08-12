import gymnasium as gym

env = gym.make("CartPole-v1")
state, info = env.reset()

print("Initial state:", state)

for step in range(5):
    action = env.action_space.sample()
    state, reward, terminated, truncated, info = env.step(action)
    print(f"Step {step+1}: action={action}, reward={reward}, new state={state}")

env.close()