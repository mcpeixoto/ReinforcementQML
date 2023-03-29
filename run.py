import pennylane as qml
import numpy as np
import gym
import torch

# Fix the seed
np.random.seed(42)
torch.manual_seed(42)


# Define the variational quantum circuit
class Classifier:
    def __init__(self, n_wires, n_layers, n_features):
        self.n_wires = n_wires
        self.n_layers = n_layers
        self.n_features = n_features

    def __call__(self, x, weights):
        # Embedding - Dense Angle Embedding
        # Divide the input in 3 parts
        assert self.n_features <= self.n_wires * 3, "Too many features"

        #print(x.shape, x)

        for i in range(self.n_features):
            if i // 3 == 0:
                qml.RX(x[i], wires=i % self.n_wires)
            elif i // 3 == 1:
                qml.RY(x[i], wires=i % self.n_wires)
            else:
                qml.RZ(x[i], wires=i % self.n_wires)

        # For every layer
        for layer in range(self.n_layers):
            W = weights[layer]

            # Define Rotations
            for i in range(self.n_wires):
                qml.Rot(W[i, 0], W[i, 1], W[i, 2], wires=i)

            # Entanglement
            if self.n_wires != 1:
                if self.n_wires > 2:
                    for i in range(self.n_wires):
                        if i == self.n_wires - 1:
                            qml.CNOT(wires=[i, 0])
                        else:
                            qml.CNOT(wires=[i, i + 1])
                else:
                    qml.CNOT(wires=[1, 0])

        # Measurement 3 qubits
        return qml.expval(qml.PauliZ(0)), qml.expval(qml.PauliZ(1)), qml.expval(qml.PauliZ(2))
    

def classifier(x, weights):
    dev = qml.device("default.qubit", wires=3)
    model = qml.QNode(Classifier(3, 2, 6), dev)

    answer = model(x, weights)
    answer = torch.tensor(answer)
    return answer


def normalize_state(state):
    """
    Normalize the state to be between -pi and pi
    
    state table:
    | state | Min | Max |
    |-------------|-----|-----|
    | Cosine of theta1 | -1 | 1 |
    | Sine of theta1 | -1 | 1 |
    | Cosine of theta2 | -1 | 1 |
    | Sine of theta2 | -1 | 1 |
    | Angular velocity of theta1 | -4*pi | 4*pi |
    | Angular velocity of theta2 | -9*pi | 9*pi |
    """

    # Normalize the observation
    state[0] = state[0] * np.pi
    state[1] = state[1] * np.pi
    state[2] = state[2] * np.pi
    state[3] = state[3] * np.pi
    state[4] = state[4] / 4
    state[5] = state[5] / 9

    return state

# Define the train loop
def train(params, env, gamma, learning_rate, regularization_strength):
    # Initialize the total reward and reset the environment
    total_reward = 0
    state = env.reset()

    # Initialize the previous state and action for the eligibility trace
    prev_state = None
    prev_action = None
    eligibility_trace = 0

    # Initialize the regularization term and the number of timesteps
    regularization_term = 0
    timestep = 0

    # Run the episode until completion
    done = False
    while not done:
        # Choose the action using the variational quantum circuit
        #print("State:", state)
        if type(state) == tuple:
            state = state[0] # torch.tensor(state[0], dtype=torch.float32)

        # Normalize the state
        state = normalize_state(state)
        # State 
        action = classifier(state, params)
        action = torch.argmax(action)


        # Take the chosen action and observe the new state and reward
        new_state, reward, done, _, _ = env.step(action)

        # Compute the reward signal
        reward = 1 - (abs(new_state[0]) + abs(new_state[1])) / 4

        # Compute the TD error and the eligibility trace
        if prev_state is not None and prev_action is not None:
            td_error = reward + gamma * classifier(new_state, params) - classifier(prev_state, params)[prev_action]
            eligibility_trace = gamma * eligibility_trace + classifier(prev_state, params)[prev_action] * (1 - gamma) * prev_action
        else:
            td_error = 0

        # Update the total reward and the state for the next iteration
        total_reward += reward
        prev_state = state
        prev_action = action
        state = new_state
        timestep += 1

        # Compute the regularization term
        regularization_term += torch.sum(torch.square(params))

        # Update the parameters using the eligibility trace and optimizer
        params -= learning_rate * td_error * eligibility_trace + regularization_strength * params

    # Compute the loss as the negative of the total reward plus the regularization term
    # (since we want to maximize the reward)
    loss = -total_reward + regularization_strength * regularization_term / timestep

    # Compute the gradient of the loss with respect to the circuit parameters
    gradient = qml.grad(loss, argnum=0)

    return loss, gradient(params)

# Initialize the environment
env = gym.make("Acrobot-v1", render_mode='human')

# Set the hyperparameters
num_episodes = 500
gamma = 0.99
learning_rate = 0.1
regularization_strength = 0#0.001

# Initialize the circuit parameters randomly
params = np.random.uniform(low=-np.pi, high=np.pi, size=(2, 6, 3))
params = torch.Tensor(params)

# Initialize the optimizer
opt = qml.AdamOptimizer(learning_rate)

# Train the model
for i in range(num_episodes):
    # Compute the loss and gradient for the current parameters
    loss, grad = train(params, env, gamma, learning_rate, regularization_strength)

    # Update the parameters using the optimizer
    params = opt.step(grad, params)

    # Print the episode number and total reward
    if (i + 1) % 10 == 0:
        total_reward = 0
        for j in range(10):
            state = env.reset()
            # Show environment
            env.render()
            done = False
            while not done:
                action = classifier(state, params)
                state, reward, done, _ = env.step(action)
                total_reward += reward
                env.render()
                print("Reward:", reward)
        print(f"Episode {i + 1}: Total reward = {total_reward / 10:.2f}")

    # Close the environment
    env.close()

"""

In this code, we first initialize the OpenAI Gym environment for the Acrobot game. Then, we set the hyperparameters, 
including the number of episodes, the discount factor gamma, the learning rate for the optimizer, and the regularization strength. 
We also randomly initialize the circuit parameters and create an Adam optimizer.

The main training loop runs for the specified number of episodes. In each episode, we use the `loss` function to compute the loss and 
gradient for the current circuit parameters and then update the parameters using the optimizer. Every 10 episodes, we print the episode 
number and the average total reward over 10 evaluation episodes.

Finally, we close the environment. Note that training a reinforcement learning algorithm with a quantum circuit can be computationally expensive, 
so this code may take some time to run.

"""