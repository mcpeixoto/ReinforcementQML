import tensorflow as tf
import gym, cirq
import numpy as np
from functools import reduce
from collections import deque
import matplotlib.pyplot as plt
import os
from os import mkdir
from os.path import join, exists
import pickle
from torch.utils.tensorboard import SummaryWriter
import hashlib
import sys
import argparse
tf.get_logger().setLevel('ERROR')

# Ignore tf warnings
import warnings
warnings.filterwarnings("ignore")


from utils import NestablePool, Rescaling, ReUploadingPQC, one_qubit_rotation, entangling_layer, generate_circuit, generate_model_Qlearning



# Defining directories
model_dir = "models"
if not exists(model_dir):
    mkdir(model_dir)




# TODO: is_classical
class CardPole():
    def __init__(self, reuploading, cx, ladder, n_layers, seed = 42, batch_size=16, lr=0.001,  n_episodes=5000,
                 max_steps=500, gamma = 0.99, show_game=False, is_classical=False,  
                 epsilon_start = 1, epsilon_decay=0.99, epsilon_min=0.01, buffer_size=10000,
                 target_update_freq=5, online_train_freq=1, win_thr = 100):

        self.bookkeeping = {}
        for key,value in locals().items():
            if key != 'self':
                setattr(self, key, value)
                if key not in ["show_game"]:
                    self.bookkeeping[key] = value

        if show_game:
            self.env = gym.make('CartPole-v1', render_mode='human')
        else:
            self.env = gym.make('CartPole-v1')

        # Set random seed
        self.env.seed(seed)
        np.random.seed(seed)
        tf.random.set_seed(seed)


        self.input_dim = 4#int(self.env.observation_space.shape[0])
        self.output_dim = 2#int(self.env.action_space.n)

        # Defining qubits and observables
        qubits = cirq.GridQubit.rect(1, self.input_dim)
        ops = [cirq.Z(q) for q in qubits]
        observables = [ops[0]*ops[1], ops[2]*ops[3]] # Z_0*Z_1 for action 0 and Z_2*Z_3 for action 1

        # Defining model
        self.model_online = generate_model_Qlearning(qubits, n_layers, cx, ladder, reuploading, self.output_dim, observables, False)
        self.model_target = generate_model_Qlearning(qubits, n_layers, cx, ladder, reuploading, self.output_dim, observables, True)

        self.model_target.set_weights(self.model_online.get_weights())

        # Init variables
        self.replay_memory = deque(maxlen=buffer_size)
        self.done = False   # Flag to indicate if the training is done
        self.win = False    # Flag to indicate if the game is won

        # Create an unique name for this run
        string = ''
        # For these variables  reuploading, cx, ladder, n_layers, seed 
        for var in ['reuploading', 'cx', 'ladder', 'n_layers']:
            string += str(var) + '_' + str(getattr(self, var)) + '_'
        
        self.name = str(hashlib.md5(string.encode()).hexdigest()) + '_' + str(seed)

        #print("STRING:", string)
        #print("SEED:", seed)
        #print("NAME:", self.name)
        

        # Saving dir
        self.save_dir = join(model_dir, self.name)
        if not exists(self.save_dir):
            mkdir(self.save_dir)
        else:
            print("WARNING: model already exists!", seed) # TODO: Add methods to resume training, load model, etc.
            raise Exception

        # Tensorboard
        self.writer = SummaryWriter(log_dir=self.save_dir)
        # Add model parameters to tensorboard
        for key, value in self.bookkeeping.items():
            self.writer.add_text(key, str(value))



    def interact_env(self, state):
        # Preprocess state
        state_array = np.array(state) 
        state = tf.convert_to_tensor([state_array])

        # Sample action
        coin = np.random.random()
        if coin > self.epsilon:
            q_vals = self.model_online([state])
            action = int(tf.argmax(q_vals[0]).numpy())
        else:
            action = np.random.choice(self.output_dim)

        # Apply sampled action in the environment, receive reward and next state
        next_state, reward, done, _ = self.env.step(action)
        
        interaction = {'state': state_array, 'action': action, 'next_state': next_state.copy(), 'reward': reward, 'done':np.float32(done)}
        
        return interaction

    @tf.function
    def Q_learning_update(self, states, actions, rewards, next_states, done):
        states = tf.convert_to_tensor(states)
        actions = tf.convert_to_tensor(actions)
        rewards = tf.convert_to_tensor(rewards)
        next_states = tf.convert_to_tensor(next_states)
        done = tf.convert_to_tensor(done)

        # Compute their target q_values and the masks on sampled actions
        future_rewards = self.model_target([next_states])
        target_q_values = rewards + (self.gamma * tf.reduce_max(future_rewards, axis=1) * (1.0 - done))
        masks = tf.one_hot(actions, self.output_dim)

        # Train the model on the states and target Q-values
        with tf.GradientTape() as tape:
            tape.watch(self.model_online.trainable_variables)
            q_values = self.model_online([states])
            q_values_masked = tf.reduce_sum(tf.multiply(q_values, masks), axis=1)
            loss = tf.keras.losses.Huber()(target_q_values, q_values_masked)

        # Backpropagation
        grads = tape.gradient(loss, self.model_online.trainable_variables)
        for optimizer, w in zip([self.optimizer_in, self.optimizer_var, self.optimizer_out], [self.w_in, self.w_var, self.w_out]):
            optimizer.apply_gradients([(grads[w], self.model_online.trainable_variables[w])])


        return loss

    def train(self):

        self.optimizer_in = tf.keras.optimizers.Adam(learning_rate=0.001, amsgrad=True)
        self.optimizer_var = tf.keras.optimizers.Adam(learning_rate=0.001, amsgrad=True)
        self.optimizer_out = tf.keras.optimizers.Adam(learning_rate=0.1, amsgrad=True)

        # Assign the model parameters to each optimizer
        self.w_in, self.w_var, self.w_out = 1, 0, 2


        self.epsilon = self.epsilon_start
        self.best_score = -np.inf


        self.episode_reward_history = []
        self.global_step = 0
        for episode in range(self.n_episodes):
            episode_reward = 0
            state = self.env.reset()
            self.episode = episode
            
            for step in range(self.max_steps):
                # Interact with env
                interaction = self.interact_env(state)
                
                # Store interaction in the replay memory
                self.replay_memory.append(interaction)
                
                state = interaction['next_state']
                episode_reward += interaction['reward']
                self.global_step += 1
                
                # Update model
                if self.global_step % self.online_train_freq == 0:
                    # Sample a batch of interactions and update Q_function
                    training_batch = np.random.choice(self.replay_memory, size=self.batch_size)
                    loss = self.Q_learning_update(np.asarray([x['state'] for x in training_batch]),
                                    np.asarray([x['action'] for x in training_batch]),
                                    np.asarray([x['reward'] for x in training_batch], dtype=np.float32),
                                    np.asarray([x['next_state'] for x in training_batch]),
                                    np.asarray([x['done'] for x in training_batch], dtype=np.float32))
                    self.writer.add_scalar('Loss', loss.numpy(), self.global_step)

                # Update target model
                if self.global_step % self.target_update_freq == 0:
                    self.model_target.set_weights(self.model_online.get_weights())
                
                # Check if the episode is finished
                if interaction['done']:
                    break

            # Average reward
            avg_reward = np.mean(self.episode_reward_history[-self.win_thr:])

            # Decay epsilon
            self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_min)
            self.writer.add_scalar('Epsilon', self.epsilon, episode)
            
            # Record reward
            self.episode_reward_history.append(episode_reward)
            self.writer.add_scalar('Episode reward', episode_reward, episode)

            # Best model
            if episode_reward >= self.best_score:
                self.best_score = episode_reward
                self.writer.add_scalar('Best score', self.best_score, episode)
                self.save()

            if avg_reward == self.max_steps:
                self.done = True
                self.win = True
                self.save() # Save the model
                # print(f"\r[INFO] Episode: {episode} | Eps: {self.epsilon:.3f} | Steps (Curr Reward): {step +1} | Best score: {self.best_score} | Avg Reward (last {self.win_thr} episodes): {avg_reward} | Win!!!", end="")
                break

        self.done = True
        self.save(save_model=False)

        # Plot the learning history of the agent:
        plt.figure(figsize=(10,5))
        plt.plot(self.episode_reward_history)
        plt.grid()
        plt.xlabel('Epsiode')
        plt.ylabel('Collected rewards')
        plt.savefig(join(self.save_dir, 'rewards.png'))
        plt.close()


    def save(self, save_model=True):
        # Save the remaining parameters to bookkeeping
        self.bookkeeping['rewards'] = self.episode_reward_history
        self.bookkeeping['done'] = self.done
        self.bookkeeping['win'] = self.win
        self.bookkeeping['episode'] = self.episode
        
        # Save the model
        if save_model:
            self.model_online.save_weights(join(self.save_dir, 'model.h5'))

        # Save the parameters
        with open(join(self.save_dir, 'params.pkl'), 'wb') as f:
            pickle.dump(self.bookkeeping, f)




if __name__ == "__main__":
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int)
    parser.add_argument('--reuploading', type=int)
    parser.add_argument('--cx', type=int)
    parser.add_argument('--ladder', type=int)
    parser.add_argument('--n_layers', type=int)
    args = parser.parse_args()

    #print("Args received:")
    #print(args)

    # Convert int to bool
    args.reuploading = bool(args.reuploading)
    args.cx = bool(args.cx)
    args.ladder = bool(args.ladder)

    # Call CardPole
    algorithm = CardPole(seed=args.seed, reuploading=args.reuploading, cx=args.cx, ladder=args.ladder, n_layers=args.n_layers)
    algorithm.train() 