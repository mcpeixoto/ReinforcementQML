import gym
import random
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import matplotlib.pyplot as plt
import numpy as np
# Quantum circuits
from circuits import VQC, exp_val_layer
# Qiskit Circuit imports
from qiskit.circuit import QuantumCircuit, QuantumRegister, Parameter, ParameterVector, ParameterExpression
from qiskit.circuit.library import TwoLocal

# Qiskit imports
import qiskit as qk
from qiskit.utils import QuantumInstance

# Qiskit Machine Learning imports
import qiskit_machine_learning as qkml
from qiskit_machine_learning.neural_networks import CircuitQNN
from qiskit_machine_learning.connectors import TorchConnector
from torch import Tensor

# Hyperparameters
LEARNING_RATE = 0.001
GAMMA = 0.99
BUFFER_SIZE = 10000
BATCH_SIZE = 16
EPSILON_START = 1.0
EPSILON_MIN = 0.01
EPSILON_DECAY = 0.99
TARGET_UPDATE_FREQ = 5
EPISODES = 200

# Q-network class
class QNetwork(nn.Module):
    def __init__(self, n_qubits, output_dim):
        super(QNetwork, self).__init__()

        self.output_dim = output_dim
        self.n_qubits = n_qubits

        # Generate the Parametrized Quantum Circuit (note the flags reuploading and reps)
        self.qc = VQC(num_qubits=self.n_qubits, reuploading=True, reps=5, measure=True, n_mes=self.output_dim)

        # Fetch the parameters from the circuit and divide them in Inputs (X) and Trainable Parameters (params)
        # The first four parameters are for the inputs
        X = list(self.qc.parameters)[:self.n_qubits]
        assert np.array(['1_Input' in str(x) for x in list(self.qc.parameters)[:self.n_qubits]]).all()

        # The remaining ones are the trainable weights of the quantum neural network
        params = list(self.qc.parameters)[self.n_qubits:]

        # Now with quasm
        qi = QuantumInstance(qk.Aer.get_backend('qasm_simulator'), shots=1000)

        # Create a Quantum Neural Network object starting from the quantum circuit defined above
        self.qnn = CircuitQNN(self.qc, input_params=X, weight_params=params, quantum_instance=qi)

        # Connect to PyTorch
        initial_weights = np.random.normal(0, 1, self.qnn.num_weights)

        quantum_nn = TorchConnector(self.qnn, initial_weights)

        exp_val = exp_val_layer(n_qubits=self.n_qubits, n_meas=self.output_dim)

        # Stack the classical and quantum layers together
        self.model = torch.nn.Sequential(quantum_nn, exp_val)

        # Plot
        self.qc.draw(output='mpl', filename='qc.png')

    def forward(self, state, no_grad=False):
        # Normalize state
        state = Tensor(state)

        if no_grad:
            with torch.no_grad():
                Q_values = self.model(state).numpy()
        else:
            Q_values = self.model(state)
        return Q_values



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
    env = gym.make('CartPole-v1')
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
        return random.randrange(network.output_dim)
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

