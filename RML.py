import random
import numpy as np
import gym
import time
import json


def train_agent(epsilon, num_epoch):
    start_epsilon = epsilon
    start_time = time.time()

    env = gym.make("FrozenLake-v1", is_slippery=False)

    q_table = np.zeros((env.observation_space.n, env.action_space.n))

    alpha = 0.1
    gamma = 0.99
    epsilon_decay = 0.999

    for _ in range(num_epoch):
        state = env.reset()[0]

        while True:
            if random.uniform(0, 1) < epsilon:
                action = env.action_space.sample()
            else:
                action = np.argmax(q_table[state])

            next_state, reward, terminated, truncated, _ = env.step(action)

            if reward == 0.0:
                reward = -10.0 if terminated else -1.0

            q_table[state][action] = (
                (1 - alpha) * q_table[state][action] +
                alpha * (reward + gamma * np.max(q_table[next_state]))
            )

            if terminated or truncated:
                break

            state = next_state

        if epsilon > 0.1:
            epsilon *= epsilon_decay

    print("Training finished.\n")

    end_time = time.time()
    execution_time = end_time - start_time
    
    print("Execution time:", execution_time, "seconds \n")
    
    env = gym.make("FrozenLake-v1", is_slippery=False, render_mode='human')
    state = env.reset()[0]
    render_start_time = time.time()
    
    while True:
        action = np.argmax(q_table[state])
        next_state, reward, done, _, _ = env.step(action)
        state = next_state

        if time.time() - render_start_time > 3:
            print("Rendering takes too long. Unable to solve with these parameters.")
            break

        if done:
            break
        
    render_time = time.time() - render_start_time

    env.close()
    
    return {
        "epsilon": start_epsilon,
        "num_epochs": num_epoch,
        "execution_time": execution_time,
        "render_time": render_time
    }

if __name__ == "__main__":
    epsilon_values = [0.1, 0.3, 0.5, 0.7, 0.9, 1]
    num_epochs_values = [30, 50, 70, 100, 300, 500, 700, 1000, 5000, 10000]

    results = []

    for epsilon in epsilon_values:
        for num_epochs in num_epochs_values:
            print(f"Training with epsilon={epsilon} and num_epochs={num_epochs}")
            results.append(train_agent(epsilon, num_epochs))
            
    with open('training_results.json', 'w') as f:
        json.dump(results, f, indent=4)

    print("Results saved to training_results.json")
