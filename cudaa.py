import torch
torch.multiprocessing.set_start_method('spawn', force=True)

import gym
import random
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import matplotlib.pyplot as plt
import numpy as np
import os
from os import mkdir
from os.path import join, exists
import hashlib
from torch import Tensor
import pennylane as qml
from torch.utils.tensorboard import SummaryWriter
import pickle
from utils import NestablePool




# Defining directories
model_dir = "models"
if not exists(model_dir):
    mkdir(model_dir)


# Q-network class
class QNetwork(nn.Module):
    def __init__(self, n_qubits, output_dim, n_layers=5, reuploading=True, device=torch.device("cuda" if torch.cuda.is_available() else "cpu")):
        super(QNetwork, self).__init__()

        # Define params
        self.output_dim = output_dim
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.reuploading = reuploading

        #self.dev = qml.device('qiskit.aer', wires=n_qubits, shots=1000)
        self.dev = qml.device('default.qubit', wires=n_qubits, shots=None)

        # Define params for optimization
        if reuploading:
            self.input_weights = torch.nn.Parameter(torch.randn(n_layers, n_qubits, requires_grad=True, device=device))
        else:
            self.input_weights = torch.nn.Parameter(torch.randn(1, n_qubits, requires_grad=True, device=device))
        self.θ = torch.nn.Parameter(torch.randn(n_layers, n_qubits, 2, requires_grad=True, device=device))
        self.output_weights = torch.nn.Parameter(torch.randn(output_dim, requires_grad=True, device=device))

        # Print shapes
        #print("Input weights shape: ", self.input_weights.shape)
        #print("θ shape: ", self.θ.shape)
        #print("Output weights shape: ", self.output_weights.shape)

        self.plot_circuit()


    def plot_circuit(self):
        fig, ax = qml.draw_mpl(qml.QNode(self.circuit, self.dev), expansion_strategy="device")(torch.randn(self.n_qubits))
        plt.savefig('circuit.png')
        return fig, ax

    def circuit(self, inputs):

        # For every layer
        for idx, layer in enumerate(range(self.n_layers)):

            # Embedding
            if self.reuploading:
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

        # Expectation values of Z of 0 and 1 qubits and 2 and 3
        return [qml.expval(qml.PauliZ(0) @ qml.PauliZ(1)), qml.expval(qml.PauliZ(2) @ qml.PauliZ(3)) ]
            

    def forward(self, x, no_grad=False):
        x = Tensor(x)
        x = torch.tanh(x)

        if no_grad:
            with torch.no_grad():
                Q_values = torch.stack([qml.QNode(self.circuit, self.dev, interface='torch', diff_method="backprop")(xi) for xi in x]).numpy()
        else:
            Q_values = torch.stack([qml.QNode(self.circuit, self.dev, interface='torch', diff_method="backprop")(xi) for xi in x])

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


class CardPole():
    def __init__(self, reuploading=True, n_layers=5, batch_size=16, lr=0.001,  n_episodes=1000, win_episodes_thr = 10,
                 max_steps=200, gamma = 0.99, show_game=False, is_classical=False, draw_circuit=False, seed = 42, 
                 epsilon_start = 1, epsilon_decay=0.99, epsilon_min=0.01, device = torch.device("cuda" if torch.cuda.is_available() else "cpu"), buffer_size=10000,
                 target_update_freq=5, online_train_freq=1 ):

        self.bookkeeping = {}
        for key,value in locals().items():
            if key != 'self':
                setattr(self, key, value)
                if key not in ["show_game", "draw_circuit"]:
                    self.bookkeeping[key] = value

        # Set seed
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        
        print("[INFO] CUDA Available: ", torch.cuda.is_available())

        if show_game:
            self.env = gym.make('CartPole-v1', render_mode='human')
        else:
            self.env = gym.make('CartPole-v1')
        input_dim = self.env.observation_space.shape[0]
        output_dim = self.env.action_space.n

        self.agent = QNetwork(input_dim, output_dim, n_layers=self.n_layers, reuploading=self.reuploading, device=self.device).float()
        self.target_network = QNetwork(input_dim, output_dim, n_layers=self.n_layers, reuploading=self.reuploading, device=self.device).float()
        self.target_network.load_state_dict(self.agent.state_dict())
        self.target_network.eval()

        self.replay_buffer = ReplayBuffer(buffer_size)

        # Two self.optimizers will be used with diferent LR, one for the input/output weights and one for the circuit parameters
        self.optimizer = optim.Adam([{'params': self.agent.output_weights, 'lr': 100*lr},
                                {'params': self.agent.θ, 'lr': lr},
                                {'params': self.agent.input_weights, 'lr': lr}])
        
        # Init variables
        self.win_cnt = 0    # Counter for the number of consecutive wins
        self.win_score = 200    # Score to reach to win the game
        self.done = False   # Flag to indicate if the training is done

        # Create an unique name for this run
        string = ''
        for key, item in self.bookkeeping.items():
            string += str(item)
        self.name = str(hashlib.md5(string.encode()).hexdigest())

         # Saving dir
        self.save_dir = join(model_dir, self.name)
        if not exists(self.save_dir):
            mkdir(self.save_dir)
        else:
            print("WARNING: model already exists!") # TODO: Add methods to resume training, load model, etc.
            raise Exception

        # Tensorboard
        self.writer = SummaryWriter(log_dir=self.save_dir)
        # Add model parameters to tensorboard
        for key, value in self.bookkeeping.items():
            self.writer.add_text(key, str(value))

    # Epsilon-greedy action
    def act(self, network, state, epsilon):
        if random.random() < epsilon:
            return random.randrange(network.output_dim)
        else:
            with torch.no_grad():
                state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
                return int(torch.argmax(network(state_tensor)).item()) # TODO


    # Make the agent learn from experience
    def learn(self, network):
        print("Learning...          ", end="\r")
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)

        states = torch.tensor(np.array(states), dtype=torch.float32)
        actions = torch.tensor(actions, dtype=torch.int64).unsqueeze(1)
        rewards = torch.tensor(rewards, dtype=torch.float32).unsqueeze(1)
        next_states = torch.tensor(np.array(next_states), dtype=torch.float32)
        dones = torch.tensor(dones, dtype=torch.bool).unsqueeze(1)

        # to GPU
        states = states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        next_states = next_states.to(self.device)
        dones = dones.to(self.device)

        current_q_values = network(states).gather(1, actions)
        next_q_values = self.target_network(next_states).max(1, keepdim=True)[0].detach() # Ignore target network gradients
        target_q_values = rewards + (self.gamma * next_q_values * (~dones))

        loss = nn.MSELoss()(current_q_values, target_q_values)

        self.optimizer.zero_grad()
        loss.backward() # Bottleneck
        self.optimizer.step()


        # Log loss
        self.writer.add_scalar('Loss (Global Step)', loss.item(), self.global_step)
        self.writer.add_scalar('Loss (Episode)', loss.item(), self.episode)


        print("Learning complete!       ", end="\r")


        # Training loop
    def train(self):
        self.rewards = []
        epsilon = self.epsilon_start
        best_score = -np.inf
        self.global_step = 0

        for episode in range(self.n_episodes):
            state, _ = self.env.reset()
            episode_reward = 0
            done = False
            self.episode = episode

            for step in range(self.win_score):
                # Play one step
                action = self.act(self.agent, state, epsilon)
                next_state, reward, done, _, _ = self.env.step(action)
                self.replay_buffer.push(state, action, reward, next_state, done)

                # Train self.agent
                if len(self.replay_buffer) > self.batch_size and step % self.online_train_freq == 0:
                    self.learn(self.agent)

                # Update target network
                if step % self.target_update_freq == 0:
                    self.target_network.load_state_dict(self.agent.state_dict())

                state = next_state
                episode_reward += reward

                self.global_step += 1

                if done:
                    break

            # Winning thr
            if step+1 >= self.win_score:
                self.win_cnt += 1
            else:
                self.win_cnt = 0

            if self.win_cnt >= self.win_episodes_thr:
                self.done = True
                self.save() # Save the model
                print(f"\r[INFO] Episode: {episode} | Eps: {self.epsilon:.3f} | Steps (Curr Reward): {step +1} | Best score: {best_score} | Win!!!")
                break

            # Saving best agent (the one that ends the fastest)
            if step > best_score:
                self.save() # Save the model
                best_score = step

                # Log the best score
                self.writer.add_scalar('Best score', best_score, episode)
            
            # Update epsilon
            epsilon = max(epsilon * self.epsilon_decay, self.epsilon_min)
            self.writer.add_scalar('Epsilon', epsilon, episode)

            # Update rewards
            self.rewards.append(episode_reward)
            self.writer.add_scalar('Reward', episode_reward, episode)

            print(f"Episode {episode}: Reward = {episode_reward}, Epsilon = {epsilon:.4f}")

        self.done = True
        self.save() # Save the model

        # Plot rewards
        plt.plot(self.rewards)
        plt.xlabel('Episode')
        plt.ylabel('Reward')
        plt.title('Training Rewards')
        plt.grid()
        plt.savefig(f"{self.save_dir}/rewards.png")


        return self.rewards
    
    def save(self, save_model=True):
        # Save the remaining parameters to bookkeeping
        self.bookkeeping['rewards'] = self.rewards
        self.bookkeeping['done'] = self.done
        
        # Save the model
        if save_model:
            torch.save(self.agent.state_dict(), join(self.save_dir, 'model.pth'))

        # Save the parameters
        with open(join(self.save_dir, 'params.pkl'), 'wb') as f:
            pickle.dump(self.bookkeeping, f)






def worker(number):
    algorithm = CardPole(show_game=False, is_classical=False, seed=number)
    algorithm.train()


if __name__ == '__main__':
    #torch.multiprocessing.set_start_method('spawn')

    # SEt multiprocessing to spawn
    #import multiprocessing
    #multiprocessing.set_start_method('spawn')

    n_runs = 5

    # Create a pool of workers
    pool = NestablePool(processes=n_runs)
   
    # Run the workers
    pool.map(worker, range(n_runs))

    # Close the pool
    pool.close()

    # Wait for the processes to finish
    pool.join()

    print("Done")