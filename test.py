import gymnasium as gym

env = gym.make("CartPole-v1")
state, info = env.reset()

print("Başlanğıc vəziyyət:", state)

for step in range(5):
    action = env.action_space.sample() 
    state, reward, terminated, truncated, info = env.step(action)
    print(f"Addım {step+1}: hərəkət={action}, mükafat={reward}, yeni vəziyyət={state}")

env.close()