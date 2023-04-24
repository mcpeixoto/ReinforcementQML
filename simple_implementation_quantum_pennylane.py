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
import pennylane as qml


# Hyperparameters
LEARNING_RATE = 0.001
GAMMA = 0.99
BUFFER_SIZE = 10000
BATCH_SIZE = 16
EPSILON_START = 1.0
EPSILON_MIN = 0.01
EPSILON_DECAY = 0.99
TARGET_UPDATE_FREQ = 10
TARGET_TRAIN_FREQ = 2
EPISODES = 200



# Q-network class
class QNetwork(nn.Module):
    def __init__(self, n_qubits, output_dim, n_layers=5, datareuploading=True):
        super(QNetwork, self).__init__()

        # Define params
        self.output_dim = output_dim
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.datareuploading = datareuploading

        #self.dev = qml.device('qiskit.aer', wires=n_qubits, shots=1000)
        self.dev = qml.device('default.qubit', wires=n_qubits, shots=1000)

        # Define params for optimization
        if datareuploading:
            self.input_weights = torch.nn.Parameter(torch.randn(n_layers, n_qubits))
        else:
            self.input_weights = torch.nn.Parameter(torch.randn(1, n_qubits))
        self.θ = torch.nn.Parameter(torch.randn(n_layers, n_qubits, 2))
        self.output_weights = torch.nn.Parameter(torch.randn(output_dim))

        # Print shapes
        print("Input weights shape: ", self.input_weights.shape)
        print("θ shape: ", self.θ.shape)
        print("Output weights shape: ", self.output_weights.shape)

        self.plot_circuit()


    def plot_circuit(self):
        fig, ax = qml.draw_mpl(qml.QNode(self.circuit, self.dev), expansion_strategy="device")(torch.randn(self.n_qubits))
        plt.savefig('circuit.png')
        return fig, ax

    def circuit(self, inputs):

        # For every layer
        for idx, layer in enumerate(range(self.n_layers)):

            # Embedding
            if self.datareuploading:
                for j in range(self.n_qubits):
                    qml.RX(self.input_weights[layer, j] * inputs[j], wires=j)
            else:
                if idx == 0:
                    for j in range(self.n_qubits):
                        qml.RX(self.input_weights[0, j] * inputs[j], wires=j)
           

            # Define Rotations in Y and Z
            for i in range(self.n_qubits):
                qml.RY(self.θ[layer, i, 0], wires=i)
                qml.RZ(self.θ[layer, i, 1], wires=i)

            # Entanglement
            for i in range(self.n_qubits):
                qml.CNOT(wires=[i, (i + 1) % self.n_qubits])
            
        # Measurement
        return [qml.expval(qml.PauliZ(i)) for i in range(self.output_dim)]
        #return qml.expval(qml.PauliZ(0))

    def forward(self, x, no_grad=False):
        x = Tensor(x)
        x = torch.tanh(x)

        if no_grad:
            with torch.no_grad():
                Q_values = torch.stack([qml.QNode(self.circuit, self.dev, interface='torch')(xi) for xi in x]).numpy()
        else:
            Q_values = torch.stack([qml.QNode(self.circuit, self.dev, interface='torch')(xi) for xi in x])

        # Output scaling
        #Q_values.shape
        #torch.Size([16, 2])
        #self.output_weights.shape
        #torch.Size([2])
        Q_values = self.output_weights * Q_values

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

            #if len(replay_buffer) > BATCH_SIZE:
            #    agent.learn(replay_buffer, target_network, optimizer)

            state = next_state
            episode_reward += reward
        
        epsilon = max(epsilon * EPSILON_DECAY, EPSILON_MIN)
        rewards.append(episode_reward)

        if  episode % TARGET_TRAIN_FREQ == 0 and  len(replay_buffer) > BATCH_SIZE:
            print("Training target network...", end="\r")
            agent.learn(replay_buffer, target_network, optimizer)
            print("Target network trained!", end=" ")

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
    loss.backward() # Bottleneck
    optimizer.step()

QNetwork.learn = learn_from_experience

if __name__ == "__main__":
    main()

