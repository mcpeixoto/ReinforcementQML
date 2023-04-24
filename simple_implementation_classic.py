import gym
import random
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import matplotlib.pyplot as plt
import numpy as np

# Hyperparameters
LEARNING_RATE = 0.001
GAMMA = 0.99
BUFFER_SIZE = 10000
BATCH_SIZE = 16
EPSILON_START = 1.0
EPSILON_MIN = 0.01
EPSILON_DECAY = 0.999
TARGET_UPDATE_FREQ = 5
EPISODES = 200

# Q-network class
class QNetwork(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, output_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        return self.fc4(x)

# Replay buffer class
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = zip(*batch)
        return state, action, reward, next_state, done

    def __len__(self):
        return len(self.buffer)

# Training loop
def train(agent, env, replay_buffer, target_network, optimizer, episodes):
    rewards = []
    epsilon = EPSILON_START

    for episode in range(episodes):
        state, _ = env.reset()
        episode_reward = 0
        done = False

        while not done:
            action = agent.act(state, epsilon)
            next_state, reward, done, _, _ = env.step(action)
            replay_buffer.push(state, action, reward, next_state, done)

            if len(replay_buffer) > BATCH_SIZE:
                agent.learn(replay_buffer, target_network, optimizer)

            state = next_state
            episode_reward += reward
            epsilon = max(epsilon * EPSILON_DECAY, EPSILON_MIN)

        rewards.append(episode_reward)

        if episode % TARGET_UPDATE_FREQ == 0:
            target_network.load_state_dict(agent.state_dict())

        print(f"Episode {episode}: Reward = {episode_reward}, Epsilon = {epsilon:.4f}")

    return rewards

# Plot rewards
def plot_rewards(rewards):
    plt.plot(rewards)
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.title('Training Rewards')
    plt.show()
    # Save the plot
    plt.savefig('rewards.png')

# Main function
def main():
    env = gym.make('CartPole-v0')
    input_dim = env.observation_space.shape[0]
    output_dim = env.action_space.n

    agent = QNetwork(input_dim, output_dim).float()
    target_network = QNetwork(input_dim, output_dim).float()
    target_network.load_state_dict(agent.state_dict())
    target_network.eval()

    replay_buffer = ReplayBuffer(BUFFER_SIZE)
    optimizer = optim.Adam(agent.parameters(), lr=LEARNING_RATE)

    rewards = train(agent, env, replay_buffer, target_network, optimizer, EPISODES)
    plot_rewards(rewards)

    env.close()

# Epsilon-greedy action selection
def epsilon_greedy_action(network, state, epsilon):
    if random.random() < epsilon:
        return random.randrange(network.fc4.out_features)
    else:
        with torch.no_grad():
            state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
            return int(torch.argmax(network(state_tensor)).item())

# Update the Q-network
QNetwork.act = epsilon_greedy_action

# Learn from experience
def learn_from_experience(self, replay_buffer, target_network, optimizer):
    states, actions, rewards, next_states, dones = replay_buffer.sample(BATCH_SIZE)

    states = torch.tensor(states, dtype=torch.float32)
    actions = torch.tensor(actions, dtype=torch.int64).unsqueeze(1)
    rewards = torch.tensor(rewards, dtype=torch.float32).unsqueeze(1)
    next_states = torch.tensor(next_states, dtype=torch.float32)
    dones = torch.tensor(dones, dtype=torch.bool).unsqueeze(1)

    current_q_values = self(states).gather(1, actions)
    next_q_values = target_network(next_states).max(1, keepdim=True)[0].detach()
    target_q_values = rewards + (GAMMA * next_q_values * (~dones))

    loss = nn.MSELoss()(current_q_values, target_q_values)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

QNetwork.learn = learn_from_experience

if __name__ == "__main__":
    main()

