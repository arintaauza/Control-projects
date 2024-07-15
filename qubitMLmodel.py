"""
    qubitMLmodel           : This is the main class that defines machine learning model for the qubit.  
"""

# Preamble
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers,optimizers,Model
import zipfile    
import os
import pickle
from QuantumModel import *
from QuantumLayers import *

class QuantumController(layers.Layer):
    """
    This class defines a custom tensorflow layer that implemements a trainable pulse generator
    """
    
    def __init__(self, T, M, n_max, pi_amp_scale, position=None,  **kwargs):
        """
        Class constructor.
        
        T             : Total time of evolution
        M             : Number of discrete time steps
        n_max         : Maximum number of control pulses in the sequence
        max_amp       : Maximum amplitude of the control pulses

        """  
        # we must call thus function for any tensorflow custom layer
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
            self.mu    = self.add_weight(name = "mu",   shape=(1, n_max, 1), dtype=tf.float32, trainable=True)  
        else:
            self.mu = tf.constant(np.reshape(position, (1,n_max,1)), dtype = tf.float32) *self.T

        self.A     = self.add_weight(name = "A",    shape=(1, n_max, 1), dtype=tf.float32, trainable=True)
        
        self.std    = self.pulse_width/6
        self.pi_amp = np.pi/( np.sqrt(np.pi)*self.std )
        self.A_max  = pi_amp_scale*self.pi_amp

        #self.mu = tf.constant( np.reshape([T*(k-0.5)/n_max for k in range(1,n_max+1)], (1,n_max,1)), dtype=tf.float32)
    def call(self, inputs):
        """
        Tensorflow method where all the calculations are done
        
        """

        # construct the signal parameters in such a way to respect the amplutide and position constraints
        temp_shape = tf.concat( [tf.shape(inputs)[0:1],tf.constant(np.array([1,1],dtype=np.int32))],0 )
        a_matrix    = tf.tile(self.a_matrix, temp_shape)
        b_matrix    = tf.tile(self.b_matrix, temp_shape)
        
        temp_shape = tf.concat( [tf.shape(inputs)[0:1],tf.constant(np.array([self.n_max,1],dtype=np.int32))],0 )     
        
        amplitude   = self.A_max*tf.tanh(self.A)
        if self.position is None:
            position    = 0.5*self.pulse_width + tf.sigmoid(self.mu)*( ( (self.T - self.n_max*self.pulse_width)/(self.n_max+1) ) - 0.5*self.pulse_width)
            position    = tf.matmul(self.a_matrix, position) + self.b_matrix
        else:
            position = self.mu

        std         = self.std * tf.ones([1, self.n_max,1], dtype=tf.float32)
       
        # combine the parameters into one tensor
        signal_parameters = tf.concat([0.5*(tf.tanh(self.A)+1)] , -1)

        # construct the signal
        #temp_shape = tf.concat( [tf.shape(inputs)[0:1],tf.constant(np.array([1,1],dtype=np.int32))],0 )     
        #time_range = tf.tile(self.time_range, temp_shape)
        tau   = [tf.reshape( tf.matmul(position[:,idx,:],  tf.ones([1,self.M]) ), (tf.shape(self.time_range)) ) for idx in range(self.n_max)]
        A     = [tf.reshape( tf.matmul(amplitude[:,idx,:], tf.ones([1,self.M]) ), (tf.shape(self.time_range)) ) for idx in range(self.n_max)]
        sigma = [tf.reshape( tf.matmul(std[:,idx,:]      , tf.ones([1,self.M]) ), (tf.shape(self.time_range)) ) for idx in range(self.n_max)]
        signal = [tf.multiply(A[idx], tf.exp( -tf.square(tf.divide(self.time_range - tau[idx], sigma[idx])) ) ) for idx in range(self.n_max)] 
        signal = tf.add_n(signal)
        
        return [signal_parameters, signal]

##############################################################################
def fidelity(y_true, y_pred):
    """
    A method for calculating fidelity for use as controller cost function
    """

    V1,V2,V3,Uc = y_pred[:,0,:], y_pred[:,1,:], y_pred[:,2,:], y_pred[:,3,:]
    W1,W2,W3,G  = y_true[:,0,:], y_true[:,1,:], y_true[:,2,:], y_true[:,3,:]
    d = 2
    f = tf.cast( tf.linalg.norm(V1 - W1) + tf.linalg.norm(V2 - W2) + tf.linalg.norm(V3 - W3), tf.float32)
    return f + 1 - (tf.square( tf.math.abs(tf.linalg.trace(tf.matmul(G, Uc, adjoint_a=True)))) /(d**2))

###############################################################################
class qubitMLmodel():
    """
    This is the main class that defines machine learning model of the qubit.
    """    
    def __init__(self, Qmodel, num_params,initial_states):
        """
        Class constructor.

        T                : Evolution time
        M                : Number of time steps
        num_params       : Number of parameters of the control pulse per each control
        dynamic_operators: A list of arrays that represent the terms of the control Hamiltonian (that depend on pulses)
        static_operators : A list of arrays that represent the terms of the drifting Hamiltonian (that are constant)
        measurement_operators: A list of arrays representing the measurement operators
        initial_states   : A list of arrays representing the initial states   
        """
        # number of axes * how many parameters
        # store the constructor arguments
        self.T                     = Qmodel.constants.T
        self.M                     = Qmodel.constants.num_steps
        self.delta_T               = self.T/self.M
        
        self.num_params            = num_params * len(Qmodel.operators.control)
        self.num_controls          = len(Qmodel.operators.control)
        # define lists for stroring the training history
        self.training_history      = []
        self.val_history           = []
        
        # define a tensorflow input layer for the normalized pulse sequence parameters
        pulse_parameters = layers.Input(shape=(None, self.num_params), name="Pulse_parameters")
    
        # define a second tensorflow input layer for the pulse sequence in time-domain
        pulse_time_domain = layers.Input(shape=(None,self.num_controls), name="Pulse_time_domain")
    
        # define the custom defined tensorflow layer that constructs the control Hamiltonian from parameters at each time step
        Hamiltonian_ctrl = HamiltonianConstruction(dynamic_operators=Qmodel.operators.control, static_operators=[Qmodel.H_d()], name="Hamiltonian")(pulse_time_domain)
    
        # define a first GRU layer that pre-processes the pulse sequence parameters 
        autocorrelation = layers.GRU(60, return_sequences=True)(pulse_parameters)
               
        # define a set of 3 different GRUs as a part of generating the parameters of each of the Vo operators  
        autocorrelation = [layers.GRU(60, return_sequences=False)(autocorrelation) for idx in range(3)]

        # define two NNs one for the producing the eigenvector parameters of the Vo operator and another one for the eigenvalues, and repeat for each Vo operator
        Vo = [VoConstruction(O = X, name="V%d"%idx_X)(
                [layers.Dense(3, activation='linear')(autocorrelation[idx_X]),  layers.Dense(1, activation='tanh')(autocorrelation[idx_X]), layers.Dense(1, activation='tanh')(autocorrelation[idx_X])]
                )for idx_X,X in enumerate(Qmodel.operators.measurement)]

        # define the custom defined tensorflow layer that constructs the final control propagtor
        Unitary     = QuantumEvolution_GB(self.T/self.M, name="Unitary")(Hamiltonian_ctrl)

        # add the custom defined tensorflow layer that calculates the measurement outcomes
        expectations = [
                [QuantumMeasurement(rho,X, name="rho%dM%d"%(idx_rho,idx_X))([Vo[idx_X],Unitary]) for idx_X, X in enumerate(Qmodel.operators.measurement)]
                for idx_rho,rho in enumerate(initial_states)]
       
        # concatenate all the measurement outcomes
        expectations = layers.Concatenate(axis=-1)(sum(expectations, [] ))
        
        # define now the tensorflow model
        self.model    = Model( inputs = [pulse_parameters, pulse_time_domain], outputs = expectations )
        
        # specify the optimizer and loss function for training 
        self.model.compile(optimizer=optimizers.Adam(lr=0.01), loss='mse')
        
        # print a summary of the model showing the layers, their connections, and the number of training parameters
        self.model.summary()

     
    def train_model(self, training_x, training_y, epochs, batch_size):
        """
        This method is for training the model given the training set
        
        training_x: A list of two  numpy arrays the first is of shape (number of examples,number of time steps, number of signal parameters), and the second is of dimensions (number of examples, number of time steps, 1)
        training_y: A numpy array that stores the meeasurement outcomes (number of examples,18).
        epochs    : The number of iterations to do the training   
        batch_size: Batch size
        """        
        # Train the model for "epochs" number of iterations using the provided training set, and store the training history
        self.training_history = self.model.fit(training_x, training_y, epochs=epochs, batch_size=batch_size,verbose=2).history["loss"] 
        
    def train_model_val(self, training_x, training_y, val_x, val_y, epochs, batch_size):
        """
        This method is for training the model given the training set and the validation set
        
        training_x: A list of two  numpy arrays the first is of shape (number of examples,number of time steps, number of signal parameters), and the second is of dimensions (number of examples, number of time steps, 1)
        training_y: A numpy array that stores the meeasurement outcomes (number of examples,18).
        epochs    : The number of iterations to do the training  
        batch_size: Batch size
        """        
        # Train the model for "epochs" number of iterations using the provided training set, and store the training history
        h  =  self.model.fit(training_x, training_y, epochs=epochs, batch_size=batch_size,verbose=2,validation_data = (val_x, val_y)) 
        self.training_history  = h.history["loss"]
        self.val_history       = h.history["val_loss"]
               
    def predict_measurements(self, testing_x):
        """
        This method is for predicting the measurement outcomes using the trained model. Usually called after training.
        
        testing_x: A list of two  numpy arrays the first is of shape (number of examples,number of time steps, number of signal parameters), and the second is of dimensions (number of examples, number of time steps, 1)
        """        
        return self.model.predict(testing_x)
    
    def predict_control_unitary(self,testing_x):
        """
        This method is for evaluating the control unitary. Usually called after training.
        
        testing_x: A list of two  numpy arrays the first is of shape (number of examples,number of time steps, number of signal parameters), and the second is of dimensions (number of examples, number of time steps, 1)
        """
        
        # define a new model that connects the input voltage and the GRU output 
        unitary_model = Model(inputs=self.model.input, outputs=self.model.get_layer('Unitary').output)
    
        # evaluate the output of this model
        return unitary_model.predict(testing_x)            
    
    def predict_Vo(self, testing_x):
        """
        This method is for predicting the Vo operators. Usally called after training.       
        testing_x: A list of two  numpy arrays the first is of shape (number of examples,number of time steps, number of signal parameters), and the second is of dimensions (number of examples, number of time steps, 1)
        200,2,2
        """
          
        # define a new model that connects the inputs to each of the Vo output layers
        Vo_model = Model(inputs=self.model.input, outputs=[self.model.get_layer(V).output for V in ["V0","V1","V2"] ] )
        
        # predict the output of the truncated model. This physically represents <U_I' O U_I>. We still need to multiply by O to get Vo = <O U_I' O U_I>
        Vo = Vo_model.predict(testing_x)
      
        return Vo
              
    def construct_controller(self, n_max, pi_amp_scale, position=None):
        """
        This method is to build a generic controller for the qubit
        
        T             : Total time of evolution
        M             : Number of discrete time steps
        n_max         : Maximum number of control pulses in the sequence
        A_max         : Maximum allowed amplitude 
        """

        #pulse_width = 0.5*T/n_max
        #sd = pulse_width/6
        #A_max = A_max* np.pi/(np.sqrt(np.pi)*sd)

        dummy_input = layers.Input(shape=(1,)) 

        # extract the part of the pre-trained qubit model & prevent it from training again
        qubit_model = Model(inputs=self.model.input, outputs=self.model.output , name='qubit_model') 
        for layer in qubit_model.layers:
            layer.trainable = False
        
        # define a custom quantum controller layer to obtain the pulse sequence
        if self.num_controls>1:
            control          = [QuantumController(self.T, self.M, n_max, pi_amp_scale,position)(dummy_input) for _ in range(self.num_controls)]
            pulse_parameters = layers.Concatenate(name="Control_Pulse_Parameters", axis=-1)([control[idx_control][0] for idx_control in range(self.num_controls)])                                                                           
            pulse_sequence   = layers.Concatenate(name="Control_Pulse_Sequence", axis=-1)(  [control[idx_control][1] for idx_control in range(self.num_controls)])
            controlled_complex = qubit_model([pulse_parameters, pulse_sequence])
        
        else:
            control = QuantumController(self.T, self.M, n_max, pi_amp_scale, position, name="Controller")(dummy_input)           
            # apply the control sequence and obtain the Vo and Uc
            controlled_complex = qubit_model(control)
        
        # define a tensorflow model for the overall controller structure
        self.controller_model = Model(inputs = dummy_input, outputs=controlled_complex)
        
        # specify the optimizer and loss function for training, with the same weight for all targets
        self.controller_model.compile(optimizer=optimizers.Adam(lr=0.01), loss="mse")

        # print a summary of the model showing the layers, their connections, and the number of training parameters
        self.controller_model.summary()
    
       
    def evaluate(self, testing_x, target_U, batch_size):
        """
        This method is for testing the mse between target and model predictions
        
        testin_x: The model input
        target_U: The target quantum gate to be designed
        
        """
        X,Y,Z = np.array([[0.,1.],[1.,0.]]), np.array([[0.,-1j],[1j,0.]]), np.array([[1.,0.],[0.,-1.]])
        initial_states = [
                         np.array([[0.5,0.5],[0.5,0.5]]), np.array([[0.5,-0.5],[-0.5,0.5]]),
                         np.array([[0.5,-0.5j],[0.5j,0.5]]),np.array([[0.5,0.5j],[-0.5j,0.5]]),
                         np.array([[1,0],[0,0]]), np.array([[0,0],[0,1]]) 
                        ]
        target = np.concatenate( [[np.trace(target_U @ rho @ target_U.conj().T @ O) for O in [X,Y,Z] ] for rho in initial_states],axis=0)
        target = np.reshape(target, (1,18))
        
        return np.average( (self.model.predict(testing_x, batch_size = batch_size, verbose=0) - np.tile(target, (testing_x[0].shape[0],1)))**2, axis=1)
    
    def train_controller(self, target_U, epochs):
        """
        This method is for training the controller to obtain some target unitary
        
        target_U: The target quantum gate to be designed
        epochs  : The number of training iterations
        """
        X,Y,Z = np.array([[0.,1.],[1.,0.]]), np.array([[0.,-1j],[1j,0.]]), np.array([[1.,0.],[0.,-1.]])
        initial_states = [
                         np.array([[0.5,0.5],[0.5,0.5]]), np.array([[0.5,-0.5],[-0.5,0.5]]),
                         np.array([[0.5,-0.5j],[0.5j,0.5]]),np.array([[0.5,0.5j],[-0.5j,0.5]]),
                         np.array([[1,0],[0,0]]), np.array([[0,0],[0,1]]) 
                        ]
        target = np.concatenate( [[np.trace(target_U @ rho @ target_U.conj().T @ O) for O in [X,Y,Z] ] for rho in initial_states],axis=0)
        target = np.reshape(target, (1,18))

        training_inputs  = np.ones((1,)) 

        # Train the model for "epochs" number of iterations using the provided training set, and store the training history
        self.controller_training_history = self.controller_model.fit(training_inputs, target, epochs=epochs, batch_size=1,verbose=0).history 
    
        # retrieve the control sequence
        if self.num_controls>1:
            control_pulses_model     = Model(inputs = self.controller_model.input, outputs = [self.controller_model.get_layer("Control_Pulse_Parameters").output, self.controller_model.get_layer("Control_Pulse_Sequence").output])
        else:
            control_pulses_model     = Model(inputs = self.controller_model.input, outputs = self.controller_model.get_layer("Controller").output)
        predicted_control_pulses = control_pulses_model.predict(training_inputs)  
        return predicted_control_pulses      

    def save_model(self, filename):
        """
        This method is to export the model to an external .mlmodel file
        filename: The name of the file (without any extensions) that stores the model.
        """
        # save all variables

        data = {'training_history':self.training_history, 
                'val_history'     :self.val_history,
                'weights'         :self.model.get_weights()
                }

        f = open(filename, 'wb')
        pickle.dump(data, f, -1)
        f.close()

    def load_model(self, filename):
        """
        This method is to import the models from an external .mlmodel file
        filename: The name of the file (without any extensions) that stores the model.
        """                     

        # load all variables

        f = open(filename, 'rb')
        data = pickle.load(f)
        f.close()          
        self.training_history  = data['training_history']
        self.val_history       = data['val_history']
        self.model.set_weights(data['weights'])
###############################################################################