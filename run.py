# General imports
import numpy as np
import matplotlib.pyplot as plt
from collections import deque

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

# PyTorch imports
import torch
from torch import Tensor
from torch.nn import MSELoss
from torch.optim import LBFGS, SGD, Adam, RMSprop

# OpenAI Gym import
import gym


from circuits import VQC, exp_val_layer
from normalize import normalize_CardPole, normalize_AcroBot

# Fix seed for reproducibility
seed = 42
np.random.seed(seed)
torch.manual_seed(seed)


# TODO: Eliminate n_qubits as a parameter
class CardPole():
    def __init__(self, reuploading=True, reps=6, batch_size=8, lr=0.01, n_episodes=1000, max_steps=500, discount_rate = 0.99, show_game=False):
        for key, value in locals().items():
            setattr(self, key, value)

        ######################
        # OpenAI Gym
        ######################

        if show_game:
            self.env = gym.make('CartPole-v1', render_mode='human')
        else:
            self.env = gym.make('CartPole-v1')

        self.input_shape = self.env.observation_space.shape
        self.n_outputs = self.env.action_space.n

        # Define qubits
        self.n_qubits = int(self.input_shape[0])

        ######################
        # CREATE PQC
        ######################

        # Generate the Parametrized Quantum Circuit (note the flags reuploading and reps)
        self.qc = VQC(num_qubits = self.n_qubits, reuploading = reuploading, reps = reps)

        # Fetch the parameters from the circuit and divide them in Inputs (X) and Trainable Parameters (params)
        # The first four parameters are for the inputs 
        X = list(self.qc.parameters)[: self.n_qubits]

        # The remaining ones are the trainable weights of the quantum neural network
        params = list(self.qc.parameters)[self.n_qubits:]

        ######################
        # PyTorch Layer
        ######################

        # Select a quantum backend to run the simulation of the quantum circuit
        # https://qiskit.org/documentation/stable/0.19/stubs/qiskit.providers.aer.StatevectorSimulator.html
        qi = QuantumInstance(qk.Aer.get_backend('statevector_simulator'), backend_options={'max_parallel_threads': 0, "max_parallel_experiments": 0, "statevector_parallel_threshold": 0})

        # Create a Quantum Neural Network object starting from the quantum circuit defined above
        self.qnn = CircuitQNN(self.qc, input_params=X, weight_params=params, quantum_instance = qi)

        # Connect to PyTorch
        initial_weights = (2*np.random.rand(self.qnn.num_weights) - 1) # Random initial weights
        quantum_nn = TorchConnector(self.qnn, initial_weights)

        exp_val = exp_val_layer(n_qubits=self.n_qubits, n_meas=self.n_outputs)

        # Stack the classical and quantum layers together 
        self.model = torch.nn.Sequential(quantum_nn, exp_val)


        ######################
        # Initialize variables
        ######################

        self.replay_memory = deque(maxlen=10000)
        self.optimizer = Adam(self.model.parameters(), lr=lr)

        self.rewards = []
        self.episodes_won = 0

        self.qc.draw(output='mpl', filename='qc.png')


    def classifier(self, state, no_grad=False):
        # Normalize state
        state = Tensor(state)
        state = normalize_CardPole(state)

        if no_grad:
            with torch.no_grad():
                Q_values = self.model(state).numpy()
        else:
            Q_values = self.model(state)
        return Q_values



    def sample_experiences(self):
        indices = np.random.randint(len(self.replay_memory), size=self.batch_size)
        batch = [self.replay_memory[index] for index in indices]
        states, actions, rewards, next_states, dones = [
            np.array([experience[field_index] for experience in batch])
            for field_index in range(5)]
        return states, actions, rewards, next_states, dones
    
    def play_one_step(self, state, epsilon):
        # Epislon-greedy policy
        if np.random.rand() < epsilon:
            action = np.random.randint(self.n_outputs)
        else:
            Q_values = self.classifier(state, no_grad=True)
            action = np.argmax(Q_values)

        next_state, reward, done, info, _ = self.env.step(action)
        self.replay_memory.append((state, action, reward, next_state, done))
        return next_state, reward, done, info

    def training_step(self):
        # Sample past experiences
        experiences = self.sample_experiences()
        states, actions, rewards, next_states, dones = experiences
        
        # Evaluate Target Q-values
        next_Q_values = self.classifier(next_states, no_grad=True)
        max_next_Q_values = np.max(next_Q_values, axis=1)

        target_Q_values = (rewards + (1 - dones) * self.discount_rate * max_next_Q_values)
        target_Q_values = target_Q_values.reshape(-1, 1)

        # Convert the actions (e.g [1, 0, 2, 1, 1..]) into one hot encoding :
        # [[0, 1, 0],
        #  [1, 0, 0],
        #  [0, 0, 1],
        #  [0, 1, 0],
        #  [0, 1, 0],
        mask = torch.nn.functional.one_hot(Tensor(actions).long(), self.n_outputs) 
        
        # Evaluate the loss
        all_Q_values = self.classifier(states)
        Q_values = torch.sum(all_Q_values * mask, dim=1, keepdims=True)
        loss = torch.mean((Q_values - Tensor(target_Q_values))**2)
        
        # Evaluate the gradients and update the parameters 
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def train(self):
        # Initialize variables
        best_score = -np.inf

        # We let the agent train for 2000 episodes
        for episode in range(self.n_episodes):
            
            # Run enviroment simulation
            obs, _ = self.env.reset()  

            for step in range(self.max_steps):
                
                # Manages the transition from exploration to exploitation
                # Based on np.exp and decay
                epsilon = max(2-np.exp((episode-100)/500), 0.01) # TODO: There's probably room to improve this
                obs, reward, done, info = self.play_one_step(obs, epsilon)
                
                if done:
                    break
            self.rewards.append(step)
            
            # Saving best agent (the one that ends the fastest)
            if step > best_score:
                torch.save(self.model.state_dict(), './best_model.pth') # Save best weights
                best_score = step
                
            
            print(f"\r[INFO] Episode: {episode} | Eps: {epsilon:.3f} | Steps (Curr Reward): {step +1} | Best score: {best_score}", end="")

            # Start training only after some exploration experiences  
            if episode > 10:
                self.training_step()



if __name__ == "__main__":
    CardPole = CardPole(show_game=False)
    CardPole.train()

    # Plot rewards
    plt.plot(CardPole.rewards)

    # Save everything
    import pickle
    with open('CardPole.pkl', 'wb') as f:
        pickle.dump(CardPole, f)
