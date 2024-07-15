from QuantumModel import *
from QuantumLayers import *

class DysonObservable(layers.Layer):
    """
    This class defines a custom tensorflow layer that takes Vx, Vy,Vz and Uc operators as inputs, and calcualate the fidelity between thm and I,I,I, and some input G repsectively. 
    """
    
    def __init__(self, Qmodel, rho, O, **kwargs):
        """
        Class constructor
        
        """   
        # we must call thus function for any tensorflow custom layer
        super(DysonObservable, self).__init__(**kwargs)
                
        self.obs = [(tf.cast(tf.expand_dims(r,0),  dtype=tf.complex128), tf.cast(tf.expand_dims(o,0),  dtype=tf.complex128)) for r,o in product(rho, O)]
        self.H_d = tf.cast(sum([a*b for a,b in zip(Qmodel.operators.drift, Qmodel.constants.drift)]), dtype = tf.complex128)
        
        self.Qmodel  = Qmodel
        self.T   = self.Qmodel.constants.T
        self.M   = self.Qmodel.constants.num_steps
        
        self.num_controls = len(Qmodel.operators.control)
        
        self.delta_t = self.T/self.M
        self.Qmodel  = Qmodel
        self.basis   = [tf.cast(b, tf.complex128) for b in [I,X,Y,Z]]
        
        g            = Qmodel.constants.g[0]
        gamma        = Qmodel.noise_parameters.constants.gamma[0]
        self.corr    = ((self.delta_t*g)**2)*tf.constant([[[np.exp(-2*gamma*abs(t1-t2)) for t1 in Qmodel.time_range] for t2 in Qmodel.time_range]], tf.complex128) 
        self.corr_gt = ((self.delta_t*g)**2)*tf.constant([[[(t2>=t1)*np.exp(-2*gamma*abs(t1-t2)) for t1 in Qmodel.time_range] for t2 in Qmodel.time_range]], tf.complex128) 
        self.corr_lt = ((self.delta_t*g)**2)*tf.constant([[[(t1>=t2)*np.exp(-2*gamma*abs(t1-t2)) for t1 in Qmodel.time_range] for t2 in Qmodel.time_range]], tf.complex128) 

    def call(self, inputs): 
        """
        This method must be defined for any custom layer, it is where the calculations are done.   
        
        inputs: a tensor of shape(num examples, num time steps, num controls) This is passed automatically by tensorflow. 
        """ 
        pulses     = [tf.tile(tf.expand_dims(tf.cast(inputs[:,:, idx_p:idx_p+1], tf.complex128), axis = -1), (1,1,self.Qmodel.dim,self.Qmodel.dim)) for idx_p in range(self.num_controls)]
        temp_shape = tf.concat( [tf.shape(inputs)[0:2],tf.constant(np.array([1,1],dtype=np.int32))],0 )   
        control    = [tf.cast(tf.tile(tf.expand_dims(tf.expand_dims(c,0),0), temp_shape), tf.complex128) for c in self.Qmodel.operators.control] 
        H_ctrl     = sum([a*b for a,b in zip(control, pulses)])
        H_d        = tf.tile( tf.expand_dims(tf.expand_dims(self.H_d,0),0), temp_shape)
        H          = self.H_d + H_ctrl
        H_1        = tf.tile( tf.expand_dims(tf.expand_dims(self.Qmodel.operators.noise[0],0),0), temp_shape)
        sigma      = [tf.tile(tf.expand_dims(tf.expand_dims(sig,0),0), temp_shape) for sig in self.basis]
     
        U          = tf.linalg.expm(-1j*H*self.delta_t) #each element is e^{-iH(delta_t)delta_t}, shape same as H
        U          = [ U[:, idx_t:idx_t+1, :,  : ] for idx_t in range(self.M)] #save U in a list
        U          = tf.concat(list(accumulate(U, lambda x, y: tf.matmul(y, x))), axis=1)
        y_a        = [0.5*tf.expand_dims(tf.linalg.trace(tf.matmul(tf.matmul(tf.matmul(U, H_1, adjoint_a=True), U),sig)), -1) for sig in sigma] #list of shape(batch, M,1)

        temp_shape = tf.concat( [tf.shape(inputs)[0:1],tf.constant(np.array([1,1],dtype=np.int32))],0 )
        C_r_pr     = tf.tile(self.corr, temp_shape)
        C_r_pr_gt  = tf.tile(self.corr_gt, temp_shape)
        C_r_pr_lt  = tf.tile(self.corr_lt, temp_shape)
        
        obs = []
        sigma      = [tf.tile(tf.expand_dims(sig,0), temp_shape) for sig in self.basis]
     
        for rho_s,O in self.obs:
            O_T = tf.matmul(tf.matmul(U[:,-1,:,:], tf.tile(O,temp_shape), adjoint_a=True), U[:,-1,:,:])
            rho = tf.tile(rho_s, temp_shape)
            T1  = tf.linalg.trace(tf.matmul(rho, O_T))
            
            T2 = 0
            T3 = 0
            T4 = 0
            
            for a, a_pr in product(range(len(sigma)), range(len(sigma))):
                I_a_apr    = tf.matmul(tf.matmul(tf.transpose(y_a[a], [0,2,1]), C_r_pr),    y_a[a_pr])
                I_a_apr_gt = tf.matmul(tf.matmul(tf.transpose(y_a[a], [0,2,1]), C_r_pr_gt), y_a[a_pr])
                I_a_apr_lt = tf.matmul(tf.matmul(tf.transpose(y_a[a], [0,2,1]), C_r_pr_lt), y_a[a_pr])
                
                T2  = T2 + tf.multiply(I_a_apr[:,0,0],    tf.linalg.trace(tf.matmul(tf.matmul(tf.matmul(sigma[a], rho), sigma[a_pr]), O_T)))
                T3  = T3 - tf.multiply(I_a_apr_gt[:,0,0], tf.linalg.trace(tf.matmul(tf.matmul(tf.matmul(sigma[a], sigma[a_pr]), rho), O_T)))
                T4  = T4 - tf.multiply(I_a_apr_lt[:,0,0], tf.linalg.trace(tf.matmul(tf.matmul(tf.matmul(rho, sigma[a]), sigma[a_pr]), O_T)))
            obs.append(tf.expand_dims(tf.math.real(T1 + T2 + T3 + T4), -1))
        return tf.concat(obs, axis=-1)
######################################################################################    
class DysonModel():
    def __init__(self, Qmodel, init_states=rho, observables=O[1:]):
        """
        class constructor
        Qmodel: Quantum Model Object
        rho: list of initial staes
        O: list of measurement operators
        """
        
        self.num_controls   = len(Qmodel.operators.control)
        self.initial_states = init_states
        self.measurement_operators = observables
        self.Qmodel = Qmodel
        
        # input layer
        pulse_time_domain = layers.Input(shape=(None,self.num_controls), name="Pulse_time_domain")
        
        # output layer
        expectations = DysonObservable(Qmodel, rho, O)(pulse_time_domain)
        
        # define now the tensorflow model
        self.model    = Model( inputs = pulse_time_domain, outputs = expectations )
        
        # specify the optimizer and loss function for training 
        self.model.compile(optimizer=optimizers.Adam(lr=0.01), loss='mse')
        
        self.model.summary()
   
    def construct_controller(self, n_max, pi_amp_scale, position=None):
        """
        This method is to build a generic controller for the qubit
        
        T             : Total time of evolution
        M             : Number of discrete time steps
        n_max         : Maximum number of control pulses in the sequence
        A_max         : Maximum allowed amplitude 
        """


        dummy_input = layers.Input(shape=(1,)) 

        # extract the part of the pre-trained qubit model & prevent it from training again
        qubit_model = Model(inputs=self.model.input, outputs=self.model.output , name='qubit_model') 
        for layer in qubit_model.layers:
            layer.trainable = False
        
        # define a custom quantum controller layer to obtain the pulse sequence
        if self.num_controls>1:
            control          = [QuantumController(self.Qmodel.constants.T, self.Qmodel.constants.num_steps, n_max, pi_amp_scale, position)(dummy_input) for _ in range(self.num_controls)]
            pulse_parameters = layers.Concatenate(name="Control_Pulse_Parameters", axis=-1)([control[idx_control][0] for idx_control in range(self.num_controls)])                                                                           
            pulse_sequence   = layers.Concatenate(name="Control_Pulse_Sequence", axis=-1)(  [control[idx_control][1] for idx_control in range(self.num_controls)])
            controlled_complex = qubit_model(pulse_sequence)
        
        else:
            control = QuantumController(self.Qmodel.constants.T, self.Qmodel.constants.num_steps, n_max, pi_amp_scale, position, name="Controller")(dummy_input)           
            # apply the control sequence and obtain the Vo and Uc
            controlled_complex = qubit_model(control[1])
        
        # define a tensorflow model for the overall controller structure
        self.controller_model = Model(inputs = dummy_input, outputs=controlled_complex)
        
        # specify the optimizer and loss function for training, with the same weight for all targets
        self.controller_model.compile(optimizer=optimizers.Adam(lr=0.01), loss="mse")

        # print a summary of the model showing the layers, their connections, and the number of training parameters
        self.controller_model.summary()
    
    def predict_measurements(self, testing_x):
        """
        This method is for predicting the measurement outcomes using the trained model. Usually called after training.
        
        testing_x: A list of two  numpy arrays the first is of shape (number of examples,number of time steps, number of signal parameters), and the second is of dimensions (number of examples, number of time steps, 1)
        """        
        return self.model.predict(testing_x)
    
    def train_controller(self, target_U, epochs):
        """
        This method is for training the controller to obtain some target unitary
        
        target_U: The target quantum gate to be designed
        epochs  : The number of training iterations
        """
        target = np.concatenate( [[np.trace(target_U @ rho @ target_U.conj().T @ O) for O in self.measurement_operators ] for rho in self.initial_states],axis=0)
        target = np.reshape(target, (1,len(self.measurement_operators)*len(self.initial_states)))

        training_inputs  = np.ones((1,)) 

        # Train the model for "epochs" number of iterations using the provided training set, and store the training history
        self.controller_training_history = self.controller_model.fit(training_inputs, target, epochs=epochs, batch_size=1,verbose=2).history 
    
        # retrieve the control sequence
        if self.num_controls>1:
            control_pulses_model     = Model(inputs = self.controller_model.input, outputs = self.controller_model.get_layer("Control_Pulse_Sequence").output)
        else:
            control_pulses_model     = Model(inputs = self.controller_model.input, outputs = self.controller_model.get_layer("Controller").output)
        predicted_control_pulses = control_pulses_model.predict(training_inputs)  
        
        return predicted_control_pulses     
