# General imports
import numpy as np
import matplotlib.pyplot as plt

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


def encode_data(inputs, num_qubits = 3):
    # inputs.shape = (6,)

    qc = qk.QuantumCircuit(num_qubits)

    # Encode data with a RX rotation
    for i in range(num_qubits): 
        qc.rx(inputs[i], i)


    return qc

def VQC(num_qubits = None, reuploading = False, reps = 2, measure = False):
    
    
    if measure:
        qr = qk.QuantumRegister(num_qubits, 'qr')
        cr = qk.ClassicalRegister(num_qubits, 'cr')
        qc = qk.QuantumCircuit(qr,cr)
    else:
        qr = qk.QuantumRegister(num_qubits, 'qr')
        qc = qk.QuantumCircuit(qr)
    
    # Define a vector containg Inputs as parameters (*not* to be optimized)
    inputs = qk.circuit.ParameterVector('x', 6)
    
    
    if not reuploading:
                
        # Encode classical input data
        qc.compose(encode_data(inputs, num_qubits = num_qubits), inplace = True)
        qc.barrier()
        
        # Variational circuit
        qc.compose(TwoLocal(num_qubits, ['ry', 'rz'], 'cz', 'circular', 
               reps=reps, insert_barriers= True, 
               skip_final_rotation_layer = True), inplace = True)
        qc.barrier()
        
        # Add final measurements
        if measure: 
            qc.measure(qr,cr)
        
    elif reuploading:
                
        # Define a vector containng variational parameters
        θ = qk.circuit.ParameterVector('θ', 2 * num_qubits * reps)
        
        # Iterate for a number of repetitions
        for rep in range(reps):

            # Encode classical input data
            qc.compose(encode_data(inputs, num_qubits = num_qubits), inplace = True)
            qc.barrier()
                
            # Variational circuit (does the same as TwoLocal from Qiskit)
            for qubit in range(num_qubits):
                qc.ry(θ[qubit + 2*num_qubits*(rep)], qubit)
                qc.rz(θ[qubit + 2*num_qubits*(rep) + num_qubits], qubit)
            qc.barrier()
                
            # Add entanglers (this code is for a circular entangler)
            qc.cz(qr[-1], qr[0])
            for qubit in range(num_qubits-1):
                qc.cz(qr[qubit], qr[qubit+1])
            qc.barrier()
                        
        # Add final measurements
        if measure: 
            qc.measure(qr,cr)
        
    return qc



class exp_val_layer(torch.nn.Module):
    def __init__(self, n_qubits = 3, n_meas = 3):
        super().__init__()
        self.n_qubits = n_qubits
        self.n_meas = n_meas

        # All possibilitiies of states given the number of qubits
        states = [bin(x)[2:].zfill(n_qubits) for x in range(2**n_qubits)]

        for i in range(n_meas):
            # Create a mask for each measurement
            setattr(self, 'mask'+str(i+1), torch.tensor([1 if x[i]=='1' else -1 for x in states], requires_grad = False))

        #self.mask1 = torch.tensor([1 if x[0]=='1' else -1 for x in states], requires_grad = False)
        #self.mask2 = torch.tensor([1 if x[1]=='1' else -1 for x in states], requires_grad = False)
        #self.mask3 = torch.tensor([1 if x[2]=='1' else -1 for x in states], requires_grad = False)


    def forward(self, x):
        """Forward step, as described above."""
        
        """
        expval_1 = self.mask1 * x
        expval_2 = self.mask2 * x
        expval_3 = self.mask3 * x
        
        # Single sample
        if len(x.shape) == 1:
            expval_1 = torch.sum(expval_1)
            expval_2 = torch.sum(expval_2)
            expval_3 = torch.sum(expval_3)
            out = torch.cat((expval_1.unsqueeze(0), expval_2.unsqueeze(0), expval_3.unsqueeze(0)), 0) # Shape: (3,)
        
        # Batch of samples
        else:
            expval_1 = torch.sum(expval_1, dim = 1, keepdim = True)
            expval_2 = torch.sum(expval_1, dim = 1, keepdim = True)
            expval_3 = torch.sum(expval_1, dim = 1, keepdim = True)
            out = torch.cat((expval_1, expval_2, expval_3), 1) # Shape: (batch_size, 3)
        """

        # General case

        # Create expval tensor
        for i in range(self.n_meas):
            setattr(self, 'expval'+str(i+1), getattr(self, 'mask'+str(i+1)) * x)

        # Single sample
        if len(x.shape) == 1:
            for i in range(self.n_meas):
                setattr(self, 'expval'+str(i+1), torch.sum(getattr(self, 'expval'+str(i+1))))
            out = torch.cat([getattr(self, 'expval'+str(i+1)).unsqueeze(0) for i in range(self.n_meas)], 0)

        # Batch of samples
        else:
            for i in range(self.n_meas):
                setattr(self, 'expval'+str(i+1), torch.sum(getattr(self, 'expval'+str(i+1)), dim = 1, keepdim = True))
            out = torch.cat([getattr(self, 'expval'+str(i+1)) for i in range(self.n_meas)], 1)
    
                
        return ((out + 1.) / 2.)