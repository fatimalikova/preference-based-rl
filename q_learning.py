import gymnasium as gym
import numpy as np

env = gym.make("FrozenLake-v1", is_slippery=False)

state_size = env.observation_space.n
action_size = env.action_space.n

q_table = np.zeros((state_size, action_size))

learning_rate = 0.8
discount_factor = 0.95
epsilon = 1.0
epsilon_decay = 0.999
episodes = 8000
epsilon_min = 0.05

rewards_per_episode = []

for episode in range(episodes):
    state, info = env.reset()
    terminated = False
    truncated = False
    total_reward = 0

    while not terminated and not truncated:
        if np.random.random() < epsilon:
            action = env.action_space.sample()  
        else:
            action = np.argmax(q_table[state])  

        next_state, reward, terminated, truncated, info = env.step(action)

        old_value = q_table[state, action]
        next_max = np.max(q_table[next_state])

        # Bellman equation: update Q-value based on reward and future estimate
        new_value = old_value + learning_rate * (reward + discount_factor * next_max - old_value)
        q_table[state, action] = new_value

        state = next_state
        total_reward += reward

    epsilon = max(epsilon_min, epsilon * epsilon_decay)
    rewards_per_episode.append(total_reward)

    if (episode + 1) % 200 == 0:
        avg_reward = np.mean(rewards_per_episode[-200:])
        print(f"Episode {episode+1}: average reward (last 200 episodes) = {avg_reward:.2f}, epsilon = {epsilon:.3f}")

print("\nLearned Q-table:")
print(q_table)

env.close()