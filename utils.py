
###########################################
# Overwritting default MultiProcess Class #
###########################################

import multiprocessing.pool

class NoDaemonProcess(multiprocessing.Process):
    @property
    def daemon(self):
        return False

    @daemon.setter
    def daemon(self, value):
        pass


class NoDaemonContext(type(multiprocessing.get_context())):
    Process = NoDaemonProcess

# We sub-class multiprocessing.pool.Pool instead of multiprocessing.Pool
# because the latter is only a wrapper function, not a proper class.
class NestablePool(multiprocessing.pool.Pool):
    def __init__(self, *args, **kwargs):
        kwargs['context'] = NoDaemonContext()
        super(NestablePool, self).__init__(*args, **kwargs)


###########################################
#        Tensorflow Quantum Utils         #
###########################################


import tensorflow as tf
import tensorflow_quantum as tfq
import cirq, sympy
import numpy as np
import matplotlib.pyplot as plt

class Rescaling(tf.keras.layers.Layer):
    def __init__(self, input_dim):
        super(Rescaling, self).__init__()
        self.input_dim = input_dim
        self.w = tf.Variable(
            initial_value=tf.ones(shape=(1,input_dim)), dtype="float32",
            trainable=True, name="obs-weights")

    def call(self, inputs):
        return tf.math.multiply((inputs+1)/2, tf.repeat(self.w,repeats=tf.shape(inputs)[0],axis=0))

class ReUploadingPQC(tf.keras.layers.Layer):
    """
    Performs the transformation (s_1, ..., s_d) -> (theta_1, ..., theta_N, lmbd[1][1]s_1, ..., lmbd[1][M]s_1,
        ......., lmbd[d][1]s_d, ..., lmbd[d][M]s_d) for d=input_dim, N=theta_dim and M=n_layers.
    An activation function from tf.keras.activations, specified by `activation` ('linear' by default) is
        then applied to all lmbd[i][j]s_i.
    All angles are finally permuted to follow the alphabetical order of their symbol names, as processed
        by the ControlledPQC.
    """

    def __init__(self, qubits, n_layers, observables, activation="linear", name="re-uploading_PQC", cx = False, ladder = False, reuploading=True):
        super(ReUploadingPQC, self).__init__(name=name)
        self.n_layers = n_layers
        self.n_qubits = len(qubits)

        circuit, theta_symbols, input_symbols = generate_circuit(qubits, n_layers, cx = cx, ladder = ladder, reuploading=reuploading)

        theta_init = tf.random_uniform_initializer(minval=0.0, maxval=np.pi)
        # Use normal distribution for the lambdas.
        lmbd_init = tf.random_normal_initializer(mean=0.0, stddev=1.0) # TODO
        self.theta = tf.Variable(
            initial_value=theta_init(shape=(1, len(theta_symbols)), dtype="float32"),
            trainable=True, name="thetas"
        )

        lmbd_init = tf.ones(shape=(self.n_qubits * self.n_layers,))
        self.lmbd = tf.Variable( # Input scaling factors.
            initial_value=lmbd_init, dtype="float32", trainable=True, name="lambdas"
        )

        # Define explicit symbol order.
        symbols = [str(symb) for symb in theta_symbols + input_symbols]
        self.indices = tf.constant([symbols.index(a) for a in sorted(symbols)])

        self.activation = activation
        self.empty_circuit = tfq.convert_to_tensor([cirq.Circuit()])
        self.computation_layer = tfq.layers.ControlledPQC(circuit, observables)        

    def call(self, inputs):
        # inputs[0] = encoding data for the state.
        batch_dim = tf.gather(tf.shape(inputs[0]), 0)
        tiled_up_circuits = tf.repeat(self.empty_circuit, repeats=batch_dim)
        tiled_up_thetas = tf.tile(self.theta, multiples=[batch_dim, 1])
        tiled_up_inputs = tf.tile(inputs[0], multiples=[1, self.n_layers])
        scaled_inputs = tf.einsum("i,ji->ji", self.lmbd, tiled_up_inputs) # Input scaling
        squashed_inputs = tf.keras.layers.Activation(self.activation)(scaled_inputs)

        joined_vars = tf.concat([tiled_up_thetas, squashed_inputs], axis=1)
        joined_vars = tf.gather(joined_vars, self.indices, axis=1)

        return self.computation_layer([tiled_up_circuits, joined_vars])

def one_qubit_rotation(qubit, symbols):
    """
    Returns Cirq gates that apply a rotation of the bloch sphere about the X,
    Y and Z axis, specified by the values in `symbols`.
    """
    return [cirq.rx(symbols[0])(qubit),
            cirq.ry(symbols[1])(qubit),
            cirq.rz(symbols[2])(qubit)]

def entangling_layer(qubits, cx = False, ladder = False):
    """
    Returns a layer of CZ entangling gates on `qubits` (arranged in a circular topology).
    """
    cz_ops = [cirq.CZ(q0, q1) for q0, q1 in zip(qubits, qubits[1:])]
    cz_ops += ([cirq.CZ(qubits[0], qubits[-1])] if len(qubits) != 2 else [])
    
    if cx:
        ''' Returns a layer of CX entangling gates on `qubits` (arranged in a circular topology).'''
        cz_ops = [cirq.CNOT(q0, q1) for q0, q1 in zip(qubits, qubits[1:])]
        cz_ops += ([cirq.CNOT(qubits[0], qubits[-1])] if len(qubits) != 2 else [])

        if ladder:
            ''' Returns a layer of CX entangling gates on `qubits` (arranged in a ladder topology).'''
            cz_ops = [cirq.CNOT(qubits[i], q1) for i in range(len(qubits)-1) for q1 in qubits[i+1:]]
    if not cx and ladder:
        ''' Returns a layer of CZ entangling gates on `qubits` (arranged in a ladder topology).'''
        cz_ops = [cirq.CZ(qubits[i], q1) for i in range(len(qubits)-1) for q1 in qubits[i+1:]]
        
    return cz_ops

def generate_circuit(qubits, n_layers, cx = False, ladder = False, reuploading=True):
    """Prepares a data re-uploading circuit on `qubits` with `n_layers` layers."""
    # Number of qubits
    n_qubits = len(qubits)

    # Sympy symbols for variational angles
    params = sympy.symbols(f'theta(0:{3*(n_layers+1)*n_qubits})')
    params = np.asarray(params).reshape((n_layers + 1, n_qubits, 3))

    # Sympy symbols for encoding angles
    if reuploading:
        inputs = sympy.symbols(f'x(0:{n_layers})'+f'_(0:{n_qubits})')
        inputs = np.asarray(inputs).reshape((n_layers, n_qubits))
    else:
        # BUG: This is giving some Runtime warnins, but it works
        inputs = sympy.symbols(f'x(0:{1})'+f'_(0:{n_qubits})')
        inputs = np.asarray(inputs).reshape((1, n_qubits))

    # Define circuit
    circuit = cirq.Circuit()
    for l in range(n_layers):
        # Variational layer
        circuit += cirq.Circuit(one_qubit_rotation(q, params[l, i]) for i, q in enumerate(qubits))
        circuit += entangling_layer(qubits, cx = cx, ladder = ladder)

        if reuploading or l== 0:
            # BUG: This is giving some Runtime warnins, but it works
            # Encoding layer
            circuit += cirq.Circuit(cirq.rx(inputs[l, i])(q) for i, q in enumerate(qubits))

    # Last varitional layer
    circuit += cirq.Circuit(one_qubit_rotation(q, params[n_layers, i]) for i,q in enumerate(qubits))

    return circuit, list(params.flat), list(inputs.flat)

def generate_model_Qlearning(qubits, n_layers, cx, ladder, reuploading, n_actions, observables, target):
    """Generates a Keras model for a data re-uploading PQC Q-function approximator."""

    input_tensor = tf.keras.Input(shape=(len(qubits), ), dtype=tf.dtypes.float32, name='input')
    re_uploading_pqc = ReUploadingPQC(qubits, n_layers, observables, activation='tanh', cx = cx, ladder = ladder, reuploading=reuploading)([input_tensor])
    process = tf.keras.Sequential([Rescaling(len(observables))], name=target*"Target"+"Q-values") # Output scaling
    Q_values = process(re_uploading_pqc)
    model = tf.keras.Model(inputs=[input_tensor], outputs=Q_values)

    return model

###########################################
#        Other Relevant Functions         #
###########################################

import itertools

def join_dicts(dict1, dict2):
    """
    Join two dictionaries together
    """
    for key, value in dict2.items():
        if key in dict1:
            dict1[key].append(value)
        else:
            dict1[key] = value
    return dict1

def GridSearch(params, fixed=None):
    """
    Given a dictionary of hyperparameters, return a
    dictionary iterator of hyperparameters.
    Example:
    params = {'n_layers': [1, 2, 3], 'n_qubits': [2, 3, 4], 'n_features': [2, 3, 4]}
    params_iterator = GridSearch(params)
    for x in params_iterator:
        print(x)
    """

    # Set all values to list
    for key, value in params.items():
        if not isinstance(value, list):
            params[key] = [value]

    # Get all keys
    keys = list(params.keys())

    # Get all values
    values = list(params.values())

    # Get all combinations
    combinations = list(itertools.product(*values))

    # Create dictionary iterator
    buff = []
    for i, combination in enumerate(combinations):
        book = dict(zip(keys, combination))
        if fixed is not None:
            book = join_dicts(book, fixed)
        #book["name"] = str(i)
        buff.append(book)

    return buff