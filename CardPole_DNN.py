import argparse
import gym
import tensorflow as tf
import numpy as np
import random
import matplotlib.pyplot as plt
from random import shuffle
from gym import wrappers
from keras.models import Sequential
from keras.layers import Dense
import pickle
from keras.models import load_model
from collections import deque
import sys
import os
from os.path import join, exists
import warnings
warnings.filterwarnings("ignore")


# Hyper-Parameters
ACTION_DIM = 2
OBSERVATIONS_DIM = 4
SAVE_DIR = 'models_DNN_fast'
if os.path.exists(SAVE_DIR) is False:
    os.makedirs(SAVE_DIR)


class ReplayBuffer:
    def __init__(self, max_size):  # Initialize the Replay Memory
        self.max_size = max_size  # Set Maximum size to buffer_size
        self.transitions = deque()  # Initialize a deque to store all samples

    def add(self, observation, action, reward, observation2):  # Function to add sample to the memory
        if len(self.transitions) > self.max_size:  # If size exceeds the max limit, remove an item
            if np.random.random() < 0.5:
                shuffle(self.transitions)  # Shuffle the Replay Memory
            self.transitions.popleft()  # Remove a Sample
        self.transitions.append((observation, action, reward, observation2))  # Add a Sample to memory

    def sample(self, count):  # Function to randomly select samples=Minibatch size(count)
        return random.sample(self.transitions, count)  # Select from transitions(Memory)

    def size(self):  # Function to keep track of Replay Memory Size
        return len(self.transitions)  # Returns the length of the Replay Memory


def get_q(model, observation):  # Function to Predict Q-Values from the Model
    np_obs = np.reshape(observation, [-1, OBSERVATIONS_DIM])  # Reshape the state
    return model.predict(np_obs)  # Query the Model for possible actions and corresponding Q-Values


def train(model, observations, targets):  # Function to Train the Model
    np_obs = np.reshape(observations, [-1, OBSERVATIONS_DIM])  # Reshape the State
    np_targets = np.reshape(targets, [-1, ACTION_DIM])  # Reshape the Target

    model.fit(np_obs, np_targets, epochs=1, verbose=0)  # Fit the model using State-Target Pairs


def predict(model, observation):  # Function to Predict Q-Values from Model
    np_obs = np.reshape(observation, [-1, OBSERVATIONS_DIM])  # Reshape the State
    return model.predict(np_obs)  # Query the Model for possible actions and corresponding Q-Values


def get_model(n_layers, lr):  # Build the Deep Q-Network
    model = Sequential()  # Type of Model

    model.add(Dense(10, input_shape=(OBSERVATIONS_DIM,), activation="relu"))

    for i in range(n_layers): 
        model.add(Dense(10, activation="relu"))
    model.add(Dense(2, activation="linear"))
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(lr=lr),
        loss="mse",
        metrics=[],
    )

    return model


def update_action(action_model, target_model, sample_transitions, gamma):  # Update the model
    random.shuffle(sample_transitions)  # Randomly Shuffle the Minibatch Samples
    batch_observations = []  # Initialize State(Observation) List
    batch_targets = []  # Initialize Target(Output Label) List

    for sample_transition in sample_transitions:  # For each sample in Minibatch
        # Separate each part of observation
        old_observation, action, reward, observation = sample_transition

        targets = np.reshape(get_q(action_model, old_observation), ACTION_DIM)
        targets[action] = reward  # Set Target Value
        if observation is not None:  # If observation is not Empty
            # Query the Model for possible actions and corresponding Q-Values
            predictions = predict(target_model, observation)
            new_action = np.argmax(predictions)  # Select the Best Action (Max Q-Value)
            # Update the Target with Future Reward Discount Factor
            targets[action] += gamma * predictions[0, new_action]

        batch_observations.append(old_observation)  # Add Old State to observations batch
        batch_targets.append(targets)  # Add target to targets batch

    # Update the model using Observations and their corresponding Targets
    train(action_model, batch_observations, batch_targets)


def save(name, model, book, save_model=True):
    # Save the model
    if save_model:
        model.save(join(SAVE_DIR, name, "model.h5"), overwrite=True)

    # Save the book-keeping variables
    with open(join(SAVE_DIR, name, "params.pkl"), "wb") as f:
        pickle.dump(book, f)




def main(n_layers, seed, batch_size = 16, lr = 0.001, n_episodes = 5000, 
         max_steps = 500, gamma = 0.99, epsilon_start = 1, epsilon_decay = 0.99, 
         epsilon_min = 0.01, buffer_size = 10000, target_update_freq = None, 
         online_train_freq = None, win_thr = 100, done = False, win = False, 
         episode = 0, is_classical=True):
    
    # Add variables for book-keeping
    book = {}
    for key, value in locals().items():
        if key != "book":
            book[key] = value
    book['win'] = False

    # Set the name of the model
    name = f"layers_{n_layers}_seed_{seed}"

    # Set random seed for reproducibility
    np.random.seed(seed)
    tf.random.set_seed(seed)

    # Initialize Variables
    iteration = 0
    epsilon = epsilon_start
    replay = ReplayBuffer(buffer_size) 
    action_model = get_model(n_layers=n_layers, lr=lr)
    env = gym.make("CartPole-v1") 
    episode_reward_history = [] 
    best_score = -np.inf

    for episode in range(n_episodes):  # For Games 0 to Maximum Games Limit
        book["episode"] = episode
        episode_reward = 0 

        # Reduce the Random Action Probability by Decay Factor
        epsilon *= epsilon_decay
        observation = env.reset()  # Reset the Environment after Each Game

        for iteration in range(max_steps):  # Timesteps
            old_observation = observation  # Store Current State

            # If generated fraction<Random Action Probability
            if np.random.random() < max(epsilon, epsilon_min):
                # Take Random Action (Explore)
                action = np.random.choice(range(ACTION_DIM))
            else:  # If generated fraction>Random Action Probability
                # Query the Model and Get Q-Values for possible actions
                q_values = get_q(action_model, observation)
                action = np.argmax(q_values)  # Select the Best Action using Q-Values received
            
            # Take the Selected Action and Observe Next State
            observation, reward, done, info = env.step(action)

            episode_reward += reward  # Add Reward to Episode Reward

            # Best Model Found
            if episode_reward > best_score:
                best_score = episode_reward
                book["best_score"] = best_score
                save(name, action_model, book, save_model=True)

            # If last win_thr episodes are 499 of reward
            if len(episode_reward_history) > win_thr and np.mean(episode_reward_history[-win_thr:]) == 499:
                win = True
                book["win"] = win
                save(name, action_model, book, save_model=True)

            if done:  # If Game Over
                book["done"] = True
                book["rewards"] = episode_reward_history

                if iteration != 499:
                    reward = -5  # Give -5 Reward for Taking Wrong Action Leading to Failure
                if iteration == 499:
                    reward = 5  # Give +5 Reward for Completing the Game Successfully
                    save(name, action_model, book, save_model=True)

                # Add the Observation to Replay Memory
                replay.add(old_observation, action, reward, None)

                break  # Break and Start a new Game

            # Add the Observation to Replay Memory
            replay.add(old_observation, action, reward, observation)

            # Save info
            save(name, action_model, book, save_model=False)

            #  Update the Deep Q-Network Model
            if replay.size() >= batch_size and np.random.random() < 0.25:
                sample_transitions = replay.sample(batch_size)
                update_action(action_model, action_model, sample_transitions, gamma)

        episode_reward_history.append(episode_reward)  # Add Episode Reward to History


def benchmark(n_layers, seed, batch_size = 16, lr = 0.001, n_episodes = 5000, 
         max_steps = 500, gamma = 0.99, epsilon_start = 1, epsilon_decay = 0.99, 
         epsilon_min = 0.01, buffer_size = 10000, target_update_freq = None, 
         online_train_freq = None, win_thr = 100, done = False, win = False, 
         episode = 0, is_classical=True):
    
    name = f"layers_{n_layers}_seed_{seed}"
    env = gym.make("CartPole-v1") 
    #############
    # BENCHMARK #
    #############

    # Load best model
    action_model = tf.keras.models.load_model(join(SAVE_DIR, name, "model.h5"))

    epsilon = 0
    rewards_over_episodes = []

    observation = env.reset()
    for episode in range(n_episodes):
        curr_epsisode_rewards = []

        state = env.reset()
        episode = episode
        

        for step in range(max_steps):
            # Query the Model and Get Q-Values for possible actions
            q_values = get_q(action_model, observation)
            action = np.argmax(q_values)  # Select the Best Action using Q-Values received
        
            # Take the Selected Action and Observe Next State
            observation, reward, done, info = env.step(action)
            
            curr_epsisode_rewards.append(reward)
            
            # Check if the episode is finished
            if done:
                break
            
        # Record reward
        rewards_over_episodes.append(curr_epsisode_rewards)

    # Save the benchmark results
    with open(join(SAVE_DIR, name, 'benchmark.pkl'), 'wb') as f:
        pickle.dump(rewards_over_episodes, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int)
    parser.add_argument("--n_layers", type=int)
    parser.add_argument('--type', type=str)
    args = parser.parse_args()

    if args.type == "benchmark":
        benchmark(n_layers=args.n_layers, seed=args.seed)
    elif args.type == "train":
        main(type=args.type, n_layers=args.n_layers, seed=args.seed)
    else:
        raise ValueError(f"Unknown type {args.type}")
