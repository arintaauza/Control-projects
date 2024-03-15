from constants import *
from QuantumModel import *
from tensorflow.keras import layers,optimizers,Model,regularizers,initializers, constraints
import tensorflow as tf
import numpy as np

"""
This module implements the machine learning-based model for the qubit. It has three classes:
    Choi                   :
    QuantumEvolution       :
    VoConstruction         : This is an internal class for constructing the Vo operators
    HamiltonianConstruction: This is an internal class for constructing Hamiltonians
    QuantumCell            : This is an internal class required for implementing time-ordered evolution
    QuantumEvolution_GB    : This is an internal class to implement time-ordered quantum evolution for graybox
    QuantumMeasurement     : This is an internal class to model coupling losses at the output.
    QuantumController      : This is an internal class that generates parameterization and time-domain representation of Gaussian control pulses
    QuantumFidelity        : This is an internal class to evaluate the fidelity between two matrices (Hilbert-Schmidt distance)
"""

#Layers will depend on the model

class Choi(layers.Layer):
    def __init__(self, Qmodel, **kwargs):
        """
        Class constructor.
        
        T             : Total time of evolution
        M             : Number of discrete time steps
        
        """  
        # we must call thus function for any tensorflow custom layer
        super(Choi, self).__init__(**kwargs)
        
        # define and store time range
        self.Qmodel     = Qmodel
        self.T          = Qmodel.constants.T
        self.M          = Qmodel.constants.num_steps
        self.K          = Qmodel.constants.num_realizations 
        
        self.static_operators = np.kron(np.eye(self.Qmodel.dim), np.array(self.Qmodel.operators.drift))
        self.dynamic_operators = np.kron(np.eye(self.Qmodel.dim), np.array(self.Qmodel.operators.control))
        self.noise_operators = np.kron(np.eye(self.Qmodel.dim), np.array(self.Qmodel.operators.noise))
        
        Id = np.expand_dims(np.eye(self.Qmodel.dim, dtype = np.complex128).flatten(), axis = 1)
        self.EPR = Id@Id.conj().T
        
        #drift
        self.H_d = tf.cast(sum([a*b for a,b in zip(self.static_operators, self.Qmodel.constants.drift)]), tf.complex128)
        
       
    def call(self, inputs):
        # pulses: (1, time, pulse_dim)
        # noise: (K, time, noise_dim)
        # returns J: (K, 4, 4) 
  
        noise, pulses = tf.cast(inputs[0], tf.complex128), tf.cast(inputs[1], tf.complex128) 
        
          
        pulses = [tf.tile(tf.expand_dims(pulses[:,:, idx_p:idx_p+1], axis = -1), (1,1,2*self.Qmodel.dim,2*self.Qmodel.dim)) for idx_p in range(pulses.shape[-1])]
        
        #control
        H_ctrl = tf.cast(sum([a*b for a,b in zip(self.dynamic_operators, pulses)]), tf.complex128) 
        
        #noise
        noise = [tf.tile(tf.expand_dims(noise[:,:,idx_n:idx_n+1], axis=-1), [1,1,2*self.Qmodel.dim,2*self.Qmodel.dim]) for idx_n in range(noise.shape[-1])]
        H_1 = tf.cast(sum([a*b for a,b in zip(self.noise_operators, noise)]), tf.complex128)
      
        #Hamiltonian
        H = self.H_d + H_ctrl + H_1
        
        #Unitary
        U = tf.linalg.expm(-1j*H*(self.T/self.M)) #each element is e^{-iH(delta_t)delta_t}, shape same as H
      
        U = [ U[:, idx_t:idx_t+1, :,  : ] for idx_t in range(self.M)] #save U in a list
        U = tf.concat(list(accumulate(U, lambda x, y: tf.matmul(y, x))), axis=1) #(K,M,2,2)

        choi_state = tf.matmul(U, tf.matmul(self.EPR, U, adjoint_b=True)) #remove all the j's, results are real

        
        return choi_state

#######################################################################################################################################


class QuantumEvolution(layers.Layer):
    def __init__(self, Qmodel, init=None, obs=None, **kwargs):
        """
        For classical noise
        Class constructor.
        Qmodel        : Quantum model
        
        T             : Total time of evolution
        M             : Number of discrete time steps
        """  
        # we must call thus function for any tensorflow custom layer
        super(QuantumEvolution, self).__init__(**kwargs)
        
        # define and store time range
        self.Qmodel     = Qmodel
        self.T          = self.Qmodel.constants.T
        self.M          = self.Qmodel.constants.num_steps
        self.K          = self.Qmodel.constants.num_realizations
        if init is not None:
            self.init = init
        else:
            self.init = rho
        if obs is not None:
            self.obs  = obs
        else:
            self.obs = O
        self.kwargs     = kwargs
       
        
        #drift
        self.H_d = tf.cast(sum([a*b for a,b in zip(self.Qmodel.operators.drift, self.Qmodel.constants.drift)]), dtype = tf.complex128)

    def call(self, inputs):
        # pulses: (K, time, 2)
        # noise: (K, time, 1)
        # returns J: (K, 18) 
        if self.kwargs["name"] == "Instantaneous":
            C = tf.tile(tf.expand_dims(tf.expand_dims(inputs,axis=-1), axis = -1), tf.constant([1 for _ in range(2*self.Qmodel.num_pulses+1)] +[2,2], dtype = tf.int32))
  
            V_O = [tf.reduce_sum(tf.multiply(C, self.Qmodel.V_O[idx_o]), axis = [idx+1 for idx in range(2*self.Qmodel.num_pulses)]) for idx_o in range(3)]
            
            if len(self.obs) == 4:
                V_O.insert(0, tf.eye(2, batch_shape = [tf.shape(inputs)[0]], dtype=tf.complex128))
            print("V_O ",V_O)

            expectation = [tf.expand_dims(tf.math.real(tf.linalg.trace(tf.matmul(V_O[idx_o], tf.expand_dims(tf.matmul(self.init[idx_r], self.obs[idx_o]), axis=0)))), axis=-1) for (idx_r,idx_o) in product(range(len(self.init)),range(len(self.obs)))]
            print("exp ", expectation)
            return tf.concat(expectation, axis=-1)
            
        else:
            noise, pulses = tf.cast(inputs[0], tf.complex128), tf.cast(inputs[1], tf.complex128) 

            #(K,M,2,2)
            #control pulses
            pulses = [tf.tile(tf.expand_dims(pulses[:,:, idx_p:idx_p+1], axis = -1), (1,1,self.Qmodel.dim,self.Qmodel.dim)) for idx_p in range(pulses.shape[-1])]
            H_ctrl = tf.cast(sum([a*b for a,b in zip(self.Qmodel.operators.control, pulses)]), dtype = tf.complex128)

            #noise
            noise = [tf.tile(tf.expand_dims(noise[:,:,idx_n:idx_n+1], axis=-1), [1,1,self.Qmodel.dim,self.Qmodel.dim]) for idx_n in range(noise.shape[-1])]
            H_1 = tf.cast(sum([a*b for a,b in zip(self.Qmodel.operators.noise, noise)]), tf.complex128)

            #Hamiltonian
            H = self.H_d + H_ctrl + H_1

            #Unitary
            delta_t = self.T/self.M
            U = tf.linalg.expm(-1j*H*delta_t) #each element is e^{-iH(delta_t)delta_t}, shape same as H
            U = [ U[:, idx_t:idx_t+1, :,  : ] for idx_t in range(self.M)] #save U in a list
            U = tf.concat(list(accumulate(U, lambda x, y: tf.matmul(y, x))), axis=1) #(K,M,2,2)

            if self.kwargs["name"] == "States":
                rho_t = [tf.expand_dims(tf.expand_dims(tf.matmul(U, tf.matmul(init, U, adjoint_b=True)), axis=1), axis=1) for init in self.init] #remove all the j's, results are real

                return tf.concat(rho_t, axis=2)
            elif self.kwargs["name"] == "Observable":
                expectations_t = [tf.expand_dims( tf.math.real( tf.linalg.trace( tf.matmul( tf.matmul(U, tf.matmul(init, U, adjoint_b=True)), obs ) )),axis=-1) for init, obs in product(self.init,self.obs)] #remove all the j's, results are real
               
                return tf.concat(expectations_t, axis=-1)



#######################################################################################################################   


class QuantumController(layers.Layer):
    """
    This class defines a custom tensorflow layer that implemements a trainable pulse generator
    """
    
    def __init__(self, T, M, n_max, pi_amp_scale, position=None, **kwargs):
        """
        Class constructor.
        
        T             : Total time of evolution
        M             : Number of discrete time steps
        n_max         : Maximum number of control pulses in the sequence
        pi_amp_scale  : Maximum amplitude of the control pulses normalized with respect to the pi pulse

        """  
        # we must call this function for any tensorflow custom layer
        super(QuantumController, self).__init__(**kwargs)
        
        # define and store time range
        self.T          = T
        self.M          = M
        self.n_max      = n_max
        self.time_range = tf.constant( np.reshape( [(0.5*T/M) + (j*T/M) for j in range(M)], (1,M,1) ) , dtype=tf.float32)
        # define the constant parmaters to shift the pulses correctly preventing overlaps
        self.pulse_width = (0.5*self.T/self.n_max)
        self.position = position
        
        self.a_matrix    = np.ones((self.n_max, self.n_max))
        self.a_matrix[np.triu_indices(self.n_max,1)] = 0
        self.a_matrix    = tf.constant(np.reshape(self.a_matrix,(1,self.n_max,self.n_max)), dtype=tf.float32)
        
        self.b_matrix    = np.reshape([idx + 0.5 for idx in range(self.n_max)], (1,self.n_max,1) ) * self.pulse_width
        self.b_matrix    = tf.constant(self.b_matrix, dtype=tf.float32)
        
        # define custom traninable weights
        if self.position is None:
            self.mu    = self.add_weight(name = "mu",   initializer = initializers.RandomNormal(mean=0.5), shape=(1, n_max, 1), dtype=tf.float32, trainable=True)  
        else:
            self.mu = tf.constant(np.reshape(position, (1,n_max,1)), dtype = tf.float32) *self.T

        self.A     = self.add_weight(name = "A",    initializer = initializers.RandomNormal() ,shape=(1, n_max, 1), dtype=tf.float32, trainable=True)
        
        self.std    = self.pulse_width/6
        self.pi_amp = np.pi/( np.sqrt(np.pi)*self.std )
        self.A_max  = pi_amp_scale*self.pi_amp
        #print("Pi pulse amplitude is %f"%self.pi_amp)
        
    def call(self, inputs):
        """
        Tensorflow method where all the calculations are done
        
        """
        
        # construct the signal parameters in such a way to respect the amplutide and position constraints
        #temp_shape = tf.concat( [tf.shape(inputs)[0:1],tf.constant(np.array([1,1],dtype=np.int32))],0 )
        #a_matrix    = tf.tile(self.a_matrix, temp_shape)
        #b_matrix    = tf.tile(self.b_matrix, temp_shape)
        
        #temp_shape = tf.concat( [tf.shape(inputs)[0:1],tf.constant(np.array([self.n_max,1],dtype=np.int32))],0 )     
        amplitude   = self.A_max*tf.tanh(self.A)
        if self.position is None:
            position    = 0.5*self.pulse_width + tf.sigmoid(self.mu)*( ( (self.T - self.n_max*self.pulse_width)/(self.n_max+1) ) - 0.5*self.pulse_width)
            position    = tf.matmul(self.a_matrix, position) + self.b_matrix
        else:
            position = self.mu

        std         = self.std * tf.ones([1, self.n_max,1], dtype=tf.float32)
       
        # combine the parameters into one tensor
        signal_parameters = tf.concat([0.5*(tf.tanh(self.A)+1), position/self.T] , -1)

        # construct the signal
        #temp_shape = tf.concat( [tf.shape(inputs)[0:1],tf.constant(np.array([1,1],dtype=np.int32))],0 )     
        #time_range = tf.tile(self.time_range, temp_shape)
        tau   = [tf.reshape( tf.matmul(position[:,idx,:],  tf.ones([1,self.M]) ), (tf.shape(self.time_range)) ) for idx in range(self.n_max)]
        A     = [tf.reshape( tf.matmul(amplitude[:,idx,:], tf.ones([1,self.M]) ), (tf.shape(self.time_range)) ) for idx in range(self.n_max)]
        sigma = [tf.reshape( tf.matmul(std[:,idx,:]      , tf.ones([1,self.M]) ), (tf.shape(self.time_range)) ) for idx in range(self.n_max)]
        signal = [tf.multiply(A[idx], tf.exp( -tf.square(tf.divide(self.time_range - tau[idx], sigma[idx])) ) ) for idx in range(self.n_max)] 
        signal = tf.add_n(signal)
        
        return [signal_parameters, signal]

####################################################################################################################### 

class QuantumControllerInstantaneous(layers.Layer):
    """
    This class defines a custom tensorflow layer that implemements a trainable pulse generator
    """

    def __init__(self, n_max, axes=3, pi_amp_scale=2, **kwargs):
        """
        Class constructor.

        n_max         : Maximum number of control pulses in the sequence
        axes          : Either 1,2 or 3 the axes where we apply control (x, or xy, or xyz)
        pi_amp_scale  : Maximum amplitude of the control pulses normalized with respect to the pi pulse

 

        """  
        # we must call thus function for any tensorflow custom layer
        super(QuantumControllerInstantaneous, self).__init__(**kwargs)

        # define and store time range
        self.n_max      = n_max
        self.axes       = axes

        # define custom traninable weights
        self.theta = [self.add_weight(name = "theta_%d"%idx_l, shape=(1,),  dtype=tf.float32, trainable=True) for idx_l in range(n_max) ]
        self.n     = [self.add_weight(name = "n_%d"%idx_l,     shape=(axes,), dtype=tf.float32, trainable=True, constraint=constraints.UnitNorm(axis=0)) for idx_l in range(n_max)]

        self.A_max  = pi_amp_scale*np.pi

    def call(self, inputs):
        """
        Tensorflow method where all the calculations are done

        """

        C_arr = tf.constant(1, dtype=tf.complex128)

        for idx_l in range(self.n_max):
            n = tf.concat([self.n[idx_l], tf.constant([0.0 for _ in range(3 - self.axes)])], axis=0) 
                          
            theta_l = self.A_max*tf.nn.sigmoid(self.theta[idx_l])
            C_l     = tf.concat([tf.cast( tf.cos(theta_l), tf.complex128)] + [-1j*tf.cast(tf.sin(theta_l)*n[idx_n], tf.complex128) for idx_n in range(3)], axis=0)

            C_arr   = tf.tensordot(C_arr, C_l, axes = 0)
       
        C_arr = tf.expand_dims( tf.tensordot(C_arr, tf.math.conj(C_arr), axes=0), axis=0)
       
        # combine the parameters into one tensor
        signal_parameters = tf.concat([tf.concat([tf.reshape(self.A_max*tf.nn.sigmoid(self.theta[idx_l]), (1,1,1)), tf.reshape(self.n[idx_l], (1,1,self.axes))], axis=2) for idx_l in range(self.n_max)], axis=1)

        return [signal_parameters,  C_arr]
    
###############################################################################
class VoConstruction(layers.Layer):
    """
    This class defines a custom tensorflow layer that takes a vector of parameters represneting eigendecompostion and reconstructs a 2x2 Hermitian traceless matrix. 
    """
    
    def __init__(self, O,  **kwargs):
        """
        Class constructor
        
        O: The observable to be measaured
        """
        # this has to be called for any tensorflow custom layer
        super(VoConstruction, self).__init__(**kwargs)
    
        self.O = tf.constant(O, dtype=tf.complex64)
        
    def call(self, x):
        """
        This method must be defined for any custom layer, it is where the calculations are done.   
        
        x: a tensor representing the inputs to the layer. This is passed automatically by tensorflow. 
        """ 
        
        # retrieve the two types of parameters from the input: 3 eigenvector parameters and 2 eigenvalue parameters
        U,x1,x2 = x
        
        # parametrize eigenvector matrix being unitary as in https://en.wikipedia.org/wiki/Unitary_matrix 
        psi   = tf.cast( U[:,0:1], tf.complex64)*1j
        theta = U[:,1:2]
        delta = tf.cast( U[:,2:], tf.complex64)*1j 
        
        # construct the first matrix
        A = tf.linalg.diag(tf.concat([tf.exp(psi), tf.exp(-psi)], -1))
        
        # construct the second matrix
        B1 = tf.expand_dims( tf.concat([tf.cos(theta), tf.sin(-theta)],-1), -1)
        B2 = tf.expand_dims( tf.concat([tf.sin(theta), tf.cos(theta)],-1), -1)
        
        B  = tf.cast( tf.concat([B1,B2],-1), tf.complex64) 
        
        # construct the third matrix
        C = tf.linalg.diag(tf.concat([tf.exp(delta), tf.exp(-delta)], -1))
        
        # multiply all three to get a Unitary (global phase shift is neglected)
        U = tf.matmul(A, tf.matmul(B,C) )
        
        # construct eigenvalue matrix such that it is traceless
        lambda1 = x1
        lambda2 = tf.multiply(x2 , 1-tf.abs(x1) )
        d = tf.concat([lambda1, lambda2], -1)*2
        d = tf.cast( tf.linalg.diag(d), tf.complex64)
        
        # construct the Hermitian tracelesss operator from its eigendecompostion
        H = tf.matmul( tf.matmul(U, d), U, adjoint_b=True)    
        
        # expand the observable operator along batch axis
        O = tf.expand_dims(self.O, 0)
        temp_shape = tf.concat( [tf.shape(U)[0:1], tf.constant(np.array([1,1],dtype=np.int32))], 0 )
        O = tf.tile(O, temp_shape)
        
        # Construct Vo operator        
        return tf.matmul(O, H)   
##############################################################################
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
        
        self.dynamic_operators = [tf.constant(op, dtype=tf.complex64) for op in dynamic_operators]
        self.static_operators  = [tf.constant(op, dtype=tf.complex64) for op in static_operators]
           
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
            # select the particular strength of the operato
            h = tf.cast(inputs[:,:,idx_op:idx_op+1] ,dtype=tf.complex64)

            # construct a tensor in the form of a row vector whose elements are [d1,d2,1,1], where d1 and d2 correspond to the
            # number of examples and number of time steps of the input
            temp_shape = tf.concat( [tf.shape(inputs)[0:2],tf.constant(np.array([1,1],dtype=np.int32))],0 )

            # add two extra dimensions for batch and time
            operator = tf.expand_dims(op,0)
            operator = tf.expand_dims(operator,0)
            
            # repeat the pauli operators along the batch and time dimensions
            operator = tf.tile(operator, temp_shape)
            
            # repeat the pulse waveform to as 2x2 matrix
            temp_shape = tf.constant(np.array([1,1,2,2],dtype=np.int32))
            h = tf.expand_dims(h,-1)
            h = tf.tile(h, temp_shape)
            
            # Now multiply each operator with its corresponding strength element-wise and add to the list of Hamiltonians
            H.append( tf.multiply(operator, h) )
       
        # loop over the strengths of all static operators
        for op in self.static_operators:          
            # construct a tensor in the form of a row vector whose elements are [d1,d2,1,1], where d1 and d2 correspond to the
            # number of examples and number of time steps of the input
            temp_shape = tf.concat( [tf.shape(inputs)[0:2],tf.constant(np.array([1,1],dtype=np.int32))],0 )

            # add two extra dimensions for batch and time
            operator = tf.expand_dims(op,0)
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
        self.delta_T= tf.constant(delta_T*-1j, dtype=tf.complex64)

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
class QuantumEvolution_GB(layers.RNN):
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
        super(QuantumEvolution_GB, self).__init__(cell, return_sequences=False,  **kwargs)
      
    def call(self, inputs):          
        """
        This method must be defined for any custom layer, it is where the calculations are done.   
        
        inputs: The tensor representing the input to the layer. This is passed automatically by tensorflow.
        """
        
        # define identity matrix with correct dimensions to be used as initial propagtor 
        dimensions = tf.shape(inputs)
        I          = tf.eye( dimensions[-1], batch_shape=[dimensions[0]], dtype=tf.complex64 )
        
        return super(QuantumEvolution_GB, self).call(inputs, initial_state=[I])         
###############################################################################    
class QuantumMeasurement(layers.Layer):
    """
    This class defines a custom tensorflow layer that takes the unitary as input, 
    and generates the measurement outcome probability as output
    """
    
    def __init__(self, initial_state, measurement_operator, **kwargs):
        """
        Class constructor
        
        initial_state       : The inital density matrix of the state before evolution.
        Measurement_operator: The measurement operator
        """          
        self.initial_state        = tf.constant(initial_state, dtype=tf.complex64)
        self.measurement_operator = tf.constant(measurement_operator, dtype=tf.complex64)
    
        # we must call thus function for any tensorflow custom layer
        super(QuantumMeasurement, self).__init__(**kwargs)
            
    def call(self, x): 
        """
        This method must be defined for any custom layer, it is where the calculations are done.   
        
        x: a tensor representing the inputs to the layer. This is passed automatically by tensorflow. 
        """ 
    
        # extract the different inputs of this layer which are the Vo and Uc
        Vo, Uc = x
        
        # construct a tensor in the form of a row vector whose elements are [d1,1,1], where d1 correspond to the
        # number of examples of the input
        temp_shape = tf.concat( [tf.shape(Uc)[0:1],tf.constant(np.array([1,1],dtype=np.int32))],0 )

        # add an extra dimension for the initial state and measurement tensors to represent batch
        initial_state        = tf.expand_dims(self.initial_state,0)
        measurement_operator = tf.expand_dims(self.measurement_operator,0)   
        
        # repeat the initial state and measurment tensors along the batch dimensions
        initial_state        = tf.tile(initial_state, temp_shape )
        measurement_operator = tf.tile(measurement_operator, temp_shape)   
        
        # evolve the initial state using the propagator provided as input
        final_state = tf.matmul(tf.matmul(Uc, initial_state), Uc, adjoint_b=True )
        
        # calculate the probability of the outcome
        expectation = tf.linalg.trace( tf.matmul( tf.matmul( Vo, final_state), measurement_operator) ) 
        
        return tf.squeeze( tf.reshape( tf.math.real(expectation), temp_shape), axis=-1 )    
###############################################################################    
class QuantumFidelity(layers.Layer):
    """
    This class defines a custom tensorflow layer that takes Vx, Vy,Vz and Uc operators as inputs, and calcualate the fidelity between thm and I,I,I, and some input G repsectively. 
    """
    
    def __init__(self, **kwargs):
        """
        Class constructor
        
        """   
        # we must call thus function for any tensorflow custom layer
        super(QuantumFidelity, self).__init__(**kwargs)
         
    def call(self, x): 
        """
        This method must be defined for any custom layer, it is where the calculations are done.   
        
        x: a tensor representing the inputs to the layer. This is passed automatically by tensorflow. 
        """ 
    
        # extract the two inputs
        U, V =  x
        
        # calculate the fidelity
        F = tf.square( tf.abs( tf.linalg.trace(tf.matmul(U, V, adjoint_a=True))  / tf.sqrt( tf.linalg.trace(tf.matmul(U,U, adjoint_a=True)) * tf.linalg.trace( tf.matmul(V,V, adjoint_a=True)) ) ))

        return F
###############################################################################
