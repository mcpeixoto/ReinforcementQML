"""
Author: Miguel Caçador Peixoto
Inspired on: https://github.com/mcpeixoto/QML-HEP
"""

# Imports
from tqdm import tqdm
import os
from os.path import basename, join
from datetime import datetime
import pickle
from functools import partial
import pennylane as qml
from pennylane import numpy as np
from pennylane.templates.embeddings import AngleEmbedding



# Base class
class BaseTrainer:
    def __init__(
        self,
        name: str,
        feature_method: str,
        n_datapoints: int,
        n_features: int,
        n_layers: int,
        max_epochs: int,
        learning_rate: float,
        random_seed: int = 42,
        check_val_every_n_epoch: int = 5,
        load_data: bool = True,
        debug: bool = False,
        write_enabled: bool = True,
        study_name: str = f"study_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}",
        results_path: str = os.path.join(os.getcwd(), "results"),
        **kwargs,
    ):
        """
        Base class for QML training

        Mandatory Args:
            - study_name (str)
                Name of the study, this will be used to identify the study
            - name (str)
                Name of the run, this will be used to identify the individual run of the study
            - feature_method (str)
                Method used to select features from the data
            - n_datapoints (int)
                Number of datapoints to use
            - n_features (int)
                Number of features to use
            - n_layers (int)
                Number of layers in the QML model
            - max_epochs (int)
                Number of epochs to train
            - learning_rate (float)
                Only required when using Adam optimizer,
                this will be the learning rate to use

        Optional Args:
            - random_seed (int) - Optional, default: 42
                Random seed for the experiment
            - check_val_every_n_epoch (int) - Optional, default: 5
                How often to check validation set
            - load_data (bool) - Optional, default: True
                Whether to load the data or not
            - debug (bool) - Optional, default: False
                Whether to print debug messages or not
            - write_enabled (bool) - Optional, default: True
                Whether to write to tensorboard or not
        """

        #######################
        # Defining Parameters #
        #######################
        # Put all parameters in a "hyper-parameter" dictionary for later use
        self.hp = {}
        for key, value in locals().items():
            if key != "self":
                setattr(self, key, value)
                # These keys can be ignored since they don't impact results
                if key not in ["load_data", "debug", "write_enabled", "kwargs"]:
                    self.hp[key] = value

        if kwargs and self.debug:
            print("[-] Variables not in use: ", kwargs)

        # Other variables
        self.best_score = None
        self.best_score_epoch = -1
        self.best_weights = None

        #######################

        if self.debug:
            print("[+] Initializing BASE trainer...")

        #############################
        # Directories, Paths & Logs #
        #############################

        # Create directories
        self.models_directory = join(results_path, f"{self.study_name}", "models")
        self.log_directory = join(results_path, f"{self.study_name}", "logs")

        for dir in [results_path, join(results_path, f"{self.study_name}"), self.models_directory, self.log_directory]:
            if not os.path.exists(dir):
                try:
                    os.mkdir(dir)
                # If it errors, this is because it already exists and is caused by parallelism
                except:
                    pass

        # Paths
        self.weights_location = join(self.models_directory, f"{self.name}.wb")
        self.log_location = join(self.log_directory, self.name)
        self.info_location = join(self.models_directory, f"{self.name}_info.pkl")

        #########################
        # QuantumML Proprieties #
        #########################

        # Define device
        self.dev = qml.device("default.qubit", wires=self.n_features)

        # Embedding
        self.embedding = partial(AngleEmbedding, wires=range(self.n_features), rotation="X")
        self.normalization = "AngleEmbedding"

    
    def init_params(self):
        return NotImplementedError("[-] Init_params not implemented")

    def circuit(self, weights, x):
        return NotImplementedError("[-] Circuit not implemented")

    def activation(self, x):
        raise NotImplementedError("[-] Activation not implemented")

    def classifier(self, weights, x):
        raise NotImplementedError("[-] Classifier not implemented")

    def train(self):
        # Needs to implement self.epoch_number
        return NotImplementedError("[-] Train not implemented")

    def load_model(self):
        # Check if model exists
        if os.path.exists(self.weights_location):
            with open(self.weights_location, "rb") as f:
                weights = pickle.load(f)
            return weights
        else:
            raise FileNotFoundError(f"[-] Model {self.study_name} - {self.name} not found")

    def plot_circuit(self):
        dummy_weights = np.random.randn(self.n_layers, self.n_features, 3, requires_grad=True)
        dummy_features = np.random.randn(1, self.n_features)
        fig, ax = qml.draw_mpl(qml.QNode(self.circuit, self.dev), expansion_strategy="device")(dummy_weights, dummy_features)
        return fig, ax