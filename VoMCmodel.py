import tensorflow as tf
from tensorflow.keras import layers,optimizers,Model
import zipfile    
import os
import pickle
from QuantumModel import *

class HamiltonianConstruction(layers.Layer):
    """
    This class defines a custom tensorflow layer that takes the Hamiltonian parameters as input, and generates the
    Hamiltonain matrix as an output at each time step for each example in the batch
    """
    def __init__(self, dynamic_operators, static_operators, **kwargs):
        """
        Class constructor 
        dynamic_operators: a list of all operators that have time-varying coefficients
        static_operators : a list of all operators that have constant coefficients
        """
        self.dynamic_operators = [tf.constant(op, dtype=tf.complex128) for op in dynamic_operators]
        self.static_operators  = [tf.constant(op, dtype=tf.complex128) for op in static_operators]
        self.dim = dynamic_operators[0].shape[-1]   
 
        # this has to be called for any tensorflow custom layer
        super(HamiltonianConstruction, self).__init__(**kwargs)
    def call(self, inputs):
        """
        This method must be defined for any custom layer, it is where the calculations are done.   
        inputs: a tensor representing the inputs to the layer. This is passed automatically by tensorflow. 
        """
 
        H = []
        # loop over the strengths of all dynamic operators
        for idx_op, op in enumerate(self.dynamic_operators):
 
            # select the particular strength of the operator
            h = tf.cast(inputs[:,:,:,idx_op:idx_op+1] ,dtype=tf.complex128)
 
            # construct a tensor in the form of a row vector whose elements are [d1,d2,d3, 1,1], where d1, d2, and d3 correspond to the
            # number of examples, number of time steps of the input, and number of realizations
            temp_shape = tf.concat( [tf.shape(inputs)[0:3],tf.constant(np.array([1,1],dtype=np.int32))],0 )
 
            # add two extra dimensions for batch, time, and realization
            operator = tf.expand_dims(op,0)
            operator = tf.expand_dims(operator,0)
            operator = tf.expand_dims(operator,0)
            # repeat the pauli operators along the batch and time dimensions
            operator = tf.tile(operator, temp_shape)
            # repeat the pulse waveform to as dxd matrix
            temp_shape = tf.constant(np.array([1,1,1,self.dim,self.dim],dtype=np.int32))
            h = tf.expand_dims(h,-1)
            h = tf.tile(h, temp_shape)
            # Now multiply each operator with its corresponding strength element-wise and add to the list of Hamiltonians
            H.append( tf.multiply(operator, h) )
        # loop over the strengths of all static operators
        for op in self.static_operators:          
            # construct a tensor in the form of a row vector whose elements are [d1,d2,d3,1,1], where d1, d2, and d2 correspond to the
            # number of examples, number of time steps of the input, and number of realizations
            temp_shape = tf.concat( [tf.shape(inputs)[0:3],tf.constant(np.array([1,1],dtype=np.int32))],0 )
 
            # add two extra dimensions for batch and time
            operator = tf.expand_dims(op,0)
            operator = tf.expand_dims(operator,0)
            operator = tf.expand_dims(operator,0)
            # repeat the pauli operators along the batch and time dimensions
            operator = tf.tile(operator, temp_shape)
            # Now add to the list of Hamiltonians
            H.append( operator )
        # now add all componenents together
        H =  tf.add_n(H)
        return H    
###############################################################################
class QuantumCell(layers.Layer):
    """
    This class defines a custom tensorflow layer that takes Hamiltonian as input, and produces one step forward propagator
    """
    def __init__(self, delta_T, **kwargs):
        """
        Class constructor.
        delta_T: time step for each propagator
        """  
        # here we define the time-step including the imaginary unit, so we can later use it directly with the expm function
        self.delta_T= tf.constant(delta_T*-1j, dtype=tf.complex128)
 
        # we must define this parameter for RNN cells
        self.state_size = [1]
        # we must call thus function for any tensorflow custom layer
        super(QuantumCell, self).__init__(**kwargs)
 
    def call(self, inputs, states):        
        """
        This method must be defined for any custom layer, it is where the calculations are done.   
        inputs: The tensor representing the input to the layer. This is passed automatically by tensorflow.
        states: The tensor representing the state of the cell. This is passed automatically by tensorflow.
        """         
        previous_output = states[0] 
        # evaluate -i*H*delta_T
        Hamiltonian = inputs * self.delta_T
        #evaluate U = expm(-i*H*delta_T)
        U = tf.linalg.expm( Hamiltonian )
        # accuamalte U to to the rest of the propagators
        new_output  = tf.matmul(U, previous_output)    
        return new_output, [new_output]
###############################################################################
class QuantumEvolution(layers.RNN):
    """
    This class defines a custom tensorflow layer that takes Hamiltonian as input, and produces the time-ordered evolution unitary as output
    """
    def __init__(self, delta_T, **kwargs):
        """
        Class constructor.
        delta_T: time step for each propagator
        """  
        # use the custom-defined QuantumCell as base class for the nodes
        cell = QuantumCell(delta_T)
 
        # we must call thus function for any tensorflow custom layer
        super(QuantumEvolution, self).__init__(cell,  **kwargs)
    def call(self, inputs):          
        """
        This method must be defined for any custom layer, it is where the calculations are done.   
        inputs: The tensor representing the input to the layer. This is passed automatically by tensorflow.
        """
        # define identity matrix with correct dimensions to be used as initial propagtor 
        dimensions = tf.shape(inputs)
        I          = tf.eye( dimensions[-1], batch_shape=[dimensions[0], dimensions[2]], dtype=tf.complex128 )
        return super(QuantumEvolution, self).call(inputs, initial_state=[I])     
###############################################################################    
class VoLayer(layers.Layer):
    """
    This class defines a custom tensorflow layer that constructs the Vo operator using the interaction picture definition
    """
    def __init__(self, O, **kwargs):
        """
        Class constructor
        O: The observable to be measaured
        """
        # this has to be called for any tensorflow custom layer
        super(VoLayer, self).__init__(**kwargs)
        self.O = tf.constant(O, dtype=tf.complex128)         
    def call(self, x):
        """
        This method must be defined for any custom layer, it is where the calculations are done.   
        x: a tensor representing the inputs to the layer. This is passed automatically by tensorflow. 
        """ 
        # retrieve the two inputs: Uc and UI
        UI,Uc = x
        UI_tilde = tf.matmul(Uc, tf.matmul(UI,Uc, adjoint_b=True) )
 
        # expand the observable operator along batch and realizations axis
        O = tf.expand_dims(self.O, 0)
        O = tf.expand_dims(O, 0)
        temp_shape = tf.concat( [tf.shape(Uc)[0:2], tf.constant(np.array([1,1],dtype=np.int32))], 0 )
        O = tf.tile(O, temp_shape)
 
        # Construct Vo operator         
        VO = tf.matmul(O, tf.matmul( tf.matmul(UI_tilde,O, adjoint_a=True), UI_tilde) )
 
        return tf.reduce_mean(VO, axis= 1)
###############################################################################
class VoMCmodel():
    """
    This is the main class that defines machine learning model of the qubit.
    """    
    def __init__(self, Qmodel, O=O[1:]):
        """
        Class constructor.
 
        """
        self.T    = Qmodel.constants.T
        self.M    = Qmodel.constants.num_steps
        self.K    = Qmodel.constants.num_realizations
        delta_T   = self.T/self.M
        self.time_range = [(0.5*self.T/self.M) + (j*self.T/self.M) for j in range(self.M)]
        self.Qmodel =  Qmodel
        # input layers
        self.num_controls = len(Qmodel.operators.control)
        pulse_time_domain = layers.Input(shape=(self.M,1, self.num_controls), name="Pulse_time_domain")
        noise_time_domain = layers.Input(shape=(self.M,self.K,len(Qmodel.operators.noise)), name="Noise_time_domain")
        # define the custom tensorflow layer that constructs the H0 part of the Hamiltonian from parameters at each time step
        Hd = sum([a*b for a,b in zip(Qmodel.operators.drift, Qmodel.constants.drift)])
        H0  = HamiltonianConstruction(dynamic_operators=Qmodel.operators.control, static_operators=[Hd], name="H0")(pulse_time_domain)
 
        # define the custom tensorflow layer that constructs the H1 part of the Hamiltonian from parameters at each time step
        H1 = HamiltonianConstruction(dynamic_operators=Qmodel.operators.noise, static_operators=[], name="H1")(noise_time_domain)
        # define the custom tensorflow layer that constructs the time-ordered evolution of H0 
        U0 = QuantumEvolution(delta_T, return_sequences=True, name="U0")(H0)
        # define Uc which is U0(T)
        Uc = layers.Lambda(lambda u0: u0[:,-1,:,:,:], name="Uc")(U0)
        # define custom tensorflow layer to calculate HI
        U0_ext = layers.Lambda(lambda x: tf.tile(x, tf.constant([1,1,self.K,1,1], dtype=tf.int32) ) )(U0)
        HI     = layers.Lambda(lambda x: tf.matmul( tf.matmul(x[0],x[1], adjoint_a=True), x[0] ), name="HI" )([U0_ext, H1])
        # define the custom defined tensorflow layer that constructs the time-ordered evolution of HI
        UI = QuantumEvolution(delta_T, return_sequences=False, name="UI")(HI)
        # construct the Vo operators
        Uc_ext = layers.Lambda(lambda x: tf.tile(x, tf.constant([1,self.K,1,1], dtype=tf.int32) ) )(Uc)        
        Vo     = [VoLayer(O, name="V%d"%idx_O)([UI,Uc_ext]) for idx_O, O in enumerate(O)]
        # define now the tensorflow model
        self.model   = Model( inputs = [pulse_time_domain, noise_time_domain], outputs = Vo )
        # print a summary of the model showing the layers and their connections
        self.model.summary()
    def predict_Vo(self, waveform):
        """
        waveform: the pulses waveform of shape (number of examples, number of time steps, number of controls)
        returns a list of Vo's, each element in the list is of shape (number of examples, dim, dim)
        """        
        noise  = np.tile(np.expand_dims(np.transpose(self.Qmodel.noise, (1,0,2)),0), (waveform.shape[0], 1, 1, 1))
        pulses = np.expand_dims(waveform, -2)
        return self.model.predict([pulses.astype(np.float64), noise.astype(np.float64)], verbose=1, batch_size = waveform.shape[0])