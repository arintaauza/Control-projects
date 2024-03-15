from QuantumModel import *
from QuantumLayers import *

class Optimizer():
    def __init__(self, Qmodel, optimizer_params, num_pulses = 5, A_max=1, init_state=None, observable=None):
        #optimizer_params : [type, learning_rate, loss]
        ### A_max change to amp_scale (25/08/23)
        
        self.Qmodel           = Qmodel
        self.A_max            = A_max
        self.num_pulses       = num_pulses
        self.init_state       = init_state 
        self.observable       = observable
        self.optimizer_params = optimizer_params
         
        ### if proc_fid : self.loss = proc_fid etc
        ### part of keras should be inside class
        
        ### define controller as a list

        
            
    ################################################################################
     #Getters
    
    @property
    def Qmodel(self):
        return self.__Qmodel
    
    @property
    def optimizer_params(self):
        return self.__optimizer_params
    
    @property
    def A_max(self):
        return self.__A_max
    
    @property
    def init_state(self):
        return self.__init_state
    
    @property
    def observable(self):
        return self.__observable
    
    ################################################################################
     #Setters     
    @Qmodel.setter
    def Qmodel(self, Qmodel):
        self.__Qmodel = Qmodel
        self.noise = self.Qmodel.noise
        self.time_range = self.Qmodel.time_range
        self.axes = len(self.Qmodel.operators.control)
    
    @optimizer_params.setter
    def optimizer_params(self, optimizer_params):
        self.__optimizer_params = objdict(optimizer_params)      
        self.learning_rate = self.optimizer_params["learning_rate"]

        if self.optimizer_params["loss"] == "matrix_norm":
            self.loss = matrix_norm
        elif self.optimizer_params["loss"] == "process_fidelity":
            self.loss = proc_fidelity
        elif self.optimizer_params["loss"] == "mse":
            self.loss = "mse"
        

        #### change this later (25/08/23)#####
        if 'position' in optimizer_params.keys():
            position = self.optimizer_params.position
        else:
            position = None
        
        if self.optimizer_params.type=="Instantaneous":
            input_dummy = layers.Input(shape=(None,1) , name = "input_dummy")
            self.controller = QuantumControllerInstantaneous(n_max=self.num_pulses, axes=self.axes, pi_amp_scale=2, name="controller")(input_dummy)
           
            out_layer   =  QuantumEvolution(self.Qmodel, name = "Instantaneous")(self.controller[1])

            self.model = Model(input_dummy, out_layer)
            self.model.compile(optimizer=optimizers.Adam(learning_rate=self.learning_rate), loss=self.loss)
                  
        else:
            noise_input   = layers.Input(shape=(None, len(self.Qmodel.operators.noise)), name="noise_input")
            self.controller    = [(QuantumController(self.Qmodel.constants.T, self.Qmodel.constants.num_steps, self.num_pulses, self.A_max, position,name="controller_%d"%(idx))(noise_input))[1] for idx in range(self.axes)]
            pulses        = layers.Concatenate(axis=-1, name="pulses")(self.controller)
     
            if self.optimizer_params["type"] == "Choi":
                choi          = Choi(self.Qmodel, name="Choi")([noise_input, pulses])
                choi_T        = layers.Lambda(lambda x: x[:,-1,:,:])(choi)
                self.model    = Model(noise_input, choi_T)
                self.model.compile(optimizer=optimizers.Adam(learning_rate=self.learning_rate), loss=self.loss)#(), metrics=[process_fidelity])
            elif self.optimizer_params["type"] == "Observable":       
                observables    = QuantumEvolution(self.Qmodel, name="Observable", init=self.rho_0, obs=self.Obs)([noise_input, pulses])
                observables_T = layers.Lambda(lambda x: x[:,-1,:])(observables)
                self.model  = Model(noise_input, observables_T)
                self.model.compile(optimizer=optimizers.Adam(self.learning_rate), loss=self.loss)#(), metrics=[process_fidelity])
            else:
                raise Exception("Wrong optimizer type.")

        self.model.summary()

    
    @A_max.setter
    def A_max(self, A_max):
        self.__A_max = A_max
        
    @init_state.setter
    def init_state(self, init_state):
        if init_state !=None:
            self.rho_0   = rho_0(self.Qmodel.dim/self.Qmodel.dim_sys, init_state)
            self.__init_state = init_state
        else: #default is rho, for one qubit
            self.rho_0 = rho_0(self.Qmodel.dim/2, rho)
            self.__init_state = rho
            
    @observable.setter
    def observable(self, observable):
        if observable != None:
            self.Obs     = Obs(self.Qmodel.dim/self.Qmodel.dim_sys, observable)
            self.__observable = observable
        else: #default is O, for one qubit
            self.Obs =  Obs(self.Qmodel.dim/2, O)
            self.__observable = O
        
    ################################################################################        
  
    
    def optimize(self, G, num_iterations, batch_size=1, verbose = 2):
        #### cast as type for target
        EPR    = tf.constant( np.array([[1],[0],[0],[1]]) @ np.array([[1],[0],[0],[1]]).conj().T, dtype=tf.complex128) # unnormalized (why quantum information theory)
        
        if self.optimizer_params["type"] == "Instantaneous":
            Obs_G = [np.reshape(np.real(np.trace(G@r@G.T.conj()@o)), (1,1)) for r,o in product(rho, O)]   
            Obs_G = np.concatenate(Obs_G, axis=-1)
            
            self.optimization_history = self.model.fit(np.array([1]),  Obs_G, epochs = num_iterations, verbose=verbose, batch_size=batch_size).history["loss"]
                        
            control_pulses_model       = Model(inputs = self.model.input, outputs = self.model.get_layer("controller").output )  ### list to retrieve controllers
            predicted_control_pulses   = control_pulses_model.predict(np.ones((1,)))  
            parameters = predicted_control_pulses[0][0]
            pulses = predicted_control_pulses[1]
            parameters = np.array([np.array([[parameters[idx][0]] + [parameters[idx][idx_vec] for idx_vec in range(2)] +[0 for _ in range(3-2)] for idx in range(5)])])
            #parameters shape is (1, num_pulses, 4) check with Pulse class
        else:    
            if self.optimizer_params["type"] == "Choi" :
                choi_G = np.reshape( np.kron(np.eye(2), G) @ EPR @ np.kron(np.eye(2), G).conj().T, (1,4,4) ).astype(np.complex128)

                #self.optimization_history = self.model.fit(np.ones((1,)),  choi_G, epochs = num_iterations, verbose=2).history["loss"]

                ### optimize method should be just to optimize, debugging from outside using performance class, fid_1, fid_2 outside
                self.optimization_history = self.model.fit(self.noise,  np.tile(choi_G, (self.Qmodel.constants.num_realizations, 1, 1)), epochs = num_iterations, verbose=verbose, batch_size=batch_size).history["loss"] ### plot this

                ### verbose =2 is slow, better use 0

                control_pulses_model       = Model(inputs = self.model.input, outputs = [self.model.get_layer("controller_%d"%(idx)).output for idx in range(len(self.controller))])
                predicted_control_pulses   = control_pulses_model.predict(np.ones((1,)))  

                parameters = np.concatenate([predicted_control_pulses[idx][0] for idx in range(self.axes)], axis = -1)
                pulses = np.concatenate([predicted_control_pulses[idx][1] for idx in range(self.axes)], axis = -1)
            elif self.optimizer_params["type"] == "Observable" :
                Obs_G = [np.reshape(np.real(np.trace(G@r@G.T.conj()@o)), (1,1)) for r,o in product(self.rho_0, self.Obs)]   
                Obs_G = np.concatenate(Obs_G, axis=-1)

            #self.optimization_history = self.model.fit(np.ones((1,)),  choi_G, epochs = num_iterations, verbose=2).history["loss"]
                self.optimization_history = self.model.fit(self.noise,  np.tile(Obs_G, (self.Qmodel.constants.num_realizations, 1)), epochs = num_iterations, verbose=verbose, batch_size=batch_size).history["loss"]

                control_pulses_model       = Model(inputs = self.model.input, outputs = [self.model.get_layer("controller_%d"%(idx)).output for idx in range(len(self.controller))])  ### list to retrieve controllers
                predicted_control_pulses   = control_pulses_model.predict(np.ones((1,)))  
                parameters = np.concatenate([predicted_control_pulses[idx][0] for idx in range(self.axes)], axis = -1)
                pulses = np.concatenate([predicted_control_pulses[idx][1] for idx in range(self.axes)], axis = -1)
    
        self.pulses = pulses
        self.pulses_params = parameters
        #return parameters, pulses

        