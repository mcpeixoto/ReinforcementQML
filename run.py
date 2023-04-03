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


# Fix seed for reproducibility
seed = 42
np.random.seed(seed)
torch.manual_seed(seed)


    
def normalize_state(state):
    """
    This function will normalize a state given from
    the AcroBot environment so it can be used by a quantum circuit
    using Angle Encoding.

    Observation Space:
    | Num | Observation                | Min                 | Max               |
    |-----|----------------------------|---------------------|-------------------|
    | 0   | Cosine of theta1           | -1                  | 1                 |
    | 1   | Sine of theta1             | -1                  | 1                 |
    | 2   | Cosine of theta2           | -1                  | 1                 |
    | 3   | Sine of theta2             | -1                  | 1                 |
    | 4   | Angular velocity of theta1 | ~ -12.567 (-4 * pi) | ~ 12.567 (4 * pi) |
    | 5   | Angular velocity of theta2 | ~ -28.274 (-9 * pi) | ~ 28.274 (9 * pi) |

    
    Source: https://www.gymlibrary.dev/environments/classic_control/acrobot/
    """


    if len(state.shape) == 1:
        assert len(state) == 6, f"Is this state from the AcroBot environment? It should have 4 elements.\n State: {state}"

        # Normalize the state to be between -pi and pi
        state[0] = state[0] * np.pi
        state[1] = state[1] * np.pi
        state[2] = state[2] * np.pi
        state[3] = state[3] * np.pi
        state[4] = state[4] / 4
        state[5] = state[5] / 9

    elif len(state.shape) == 2:
        assert state.shape[1] == 6, f"Is this state from the AcroBot environment? It should have 4 elements.\n State: {state}"

        # Normalize the state to be between -pi and pi
        state[:,0] = state[:,0] * np.pi
        state[:,1] = state[:,1] * np.pi
        state[:,2] = state[:,2] * np.pi
        state[:,3] = state[:,3] * np.pi
        state[:,4] = state[:,4] / 4
        state[:,5] = state[:,5] / 9

    return state



# TODO: Eliminate n_qubits as a parameter
class Acrobot():
    def __init__(self, reuploading=True, reps=6, batch_size=16, lr=0.01, n_episodes=1000, max_steps=250, discount_rate = 0.99, show_game=False):
        for key, value in locals().items():
            setattr(self, key, value)

        ######################
        # OpenAI Gym
        ######################

        if show_game:
            self.env = gym.make('Acrobot-v1', render_mode='human')
        else:
            self.env = gym.make('Acrobot-v1')

        self.input_shape = self.env.observation_space.shape
        self.n_outputs = self.env.action_space.n

        # Define qubits
        self.n_qubits = int(self.input_shape[0])

        ######################
        # CREATE PQC
        ######################

        # Generate the Parametrized Quantum Circuit (note the flags reuploading and reps)
        self.qc = VQC(num_qubits = self.n_qubits, reuploading = False)

        # Fetch the parameters from the circuit and divide them in Inputs (X) and Trainable Parameters (params)
        # The first four parameters are for the inputs 
        X = list(self.qc.parameters)[: self.n_qubits]

        # The remaining ones are the trainable weights of the quantum neural network
        params = list(self.qc.parameters)[self.n_qubits:]

        ######################
        # PyTorch Layer
        ######################

        # Select a quantum backend to run the simulation of the quantum circuit
        qi = QuantumInstance(qk.Aer.get_backend('statevector_simulator'))

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


    def classifier(self, state, no_grad=False):
        # Normalize state
        state = Tensor(state)
        state = normalize_state(state)

        if no_grad:
            with torch.no_grad():
                Q_values = self.model(Tensor(state)).numpy()
        else:
            Q_values = self.model(Tensor(state)).detatch().numpy()
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
            action = np.argmax(Q_values[0])

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
        # TODO: IMPROVE
        rewards = [] 
        best_score = self.max_steps

        # We let the agent train for 2000 episodes
        for episode in range(self.n_episodes):
            
            # Run enviroment simulation
            obs, _ = self.env.reset()  

            for step in range(self.max_steps):
                
                # Manages the transition from exploration to exploitation
                # Based on np.exp and decay
                epsilon = max(2-np.exp(episode/20), 0.01) # TODO: There's probably room to improve this
                obs, reward, done, info = self.play_one_step(obs, epsilon)
                if done:
                    break
            rewards.append(step)
            
            # Saving best agent (the one that ends the fastest)
            if step <= best_score:
                torch.save(self.model.state_dict(), './best_model.pth') # Save best weights
                best_score = step
                
            print("\rEpisode: {}, Steps : {}, Eps: {:.3f}, epsilon: {:.3f}, Best score: {}".format(episode, step, epsilon, epsilon, best_score), end="")
            
            # Start training only after some exploration experiences  
            if episode > 5:
                self.training_step()



if __name__ == "__main__":
    acrobot = Acrobot(show_game=True)
    acrobot.train()