from QuantumModel import *
from QuantumLayers import *
import numpy as np
import tensorflow as tf

class Simulator:   
    def __init__(self, Qmodel, sim_type, init_states = None, observables = None):
        """
        sim_type = ["Choi", "Observable_t", "Observable_T", "States_t", "State_T", "State_Observable_t", "State_Observable_T"] #classical noise  
        
        Attributes:
        Qmodel    : QuantumModel object
        sim_type  : type of simulator
        model     : keras model for simulator
        noise     : noise, taken from Qmodel
        data      : list of length max 2, data[0]:state, data[1]:expectation
        rho_0     : list of initial states (init_sys \otimes Zero)
        Obs       : list of observables (sigma_sys \otimes I)
        
        Private attributes:
        T
        M
        K
        
        """
        
        self.Qmodel         = Qmodel            
        self.init_states    = init_states
        self.observables    = observables
        self.sim_type       = sim_type         
            
            
    ################################################################################
     #Getters
    @property
    def sim_type(self):
        return self.__sim_type
    
    @property
    def Qmodel(self):
        return self.__Qmodel
    
    #@property
    #def targets(self):
    #    return self.__targets
    
    @property
    def init_states(self):
        return self.__init_states
    
    @property
    def observables(self):
        return self.__observables
    ################################################################################
     #Setters     
    @sim_type.setter
    def sim_type(self, sim_type): 
        if sim_type == "Instantaneous":
            input_shape = tuple([4 for _ in range(2*self.Qmodel.num_pulses)])
            input_C     = layers.Input(shape=input_shape, name="input_C", dtype = tf.complex128)
            out_layer   = QuantumEvolution(Qmodel = self.Qmodel, name="Instantaneous", init=self.rho_0, obs=self.Obs)(input_C)
            #out_layer   = layers.Lambda(lambda C:tf.concat([tf.math.real(tf.linalg.trace(tf.matmul(tf.reduce_sum(tf.multiply(tf.tile(tf.expand_dims(tf.expand_dims(C,axis=-1), axis = -1), tf.constant([1 for _ in range(2*self.Qmodel.num_pulses+1)] +[2,2], dtype = tf.int32)), self.Qmodel.V_O[idx_o]), axis = [idx+1 for idx in range(2*self.Qmodel.num_pulses)]), tf.expand_dims(tf.matmul(rho[idx_r], O[idx_o]), axis=0)))) for (idx_r,idx_o) in product(range(6),range(3))], axis = -1))(input_C)
            self.model = Model(input_C, out_layer)
        else:   
            noise_input   = layers.Input(shape=(None, len(self.Qmodel.operators.noise)), name="noise_input")
            pulses        = layers.Input(shape=(None,len(self.Qmodel.operators.control)), name="control_input")
        
            if sim_type == "Choi":
                states  = Choi(Qmodel = self.Qmodel, name="Choi")([noise_input, pulses])
                states  = layers.Lambda(lambda x: x[:,-1,:,:])(states)
            elif sim_type == "Choi_t":
                states  = Choi(Qmodel = self.Qmodel, name="Choi")([noise_input, pulses])
            elif sim_type == "State_t" or sim_type == "State_Observable_t":
                states  = QuantumEvolution(Qmodel = self.Qmodel, name="States", init=self.rho_0)([noise_input, pulses])
            elif sim_type == "State_T" or sim_type == "State_Observable_T":
                states  = QuantumEvolution(Qmodel = self.Qmodel, name="States",init=self.rho_0)([noise_input, pulses])
                states  = layers.Lambda(lambda x: x[:,:,:,-1,:,:])(states)
            elif sim_type == "Observable_t":
                states  = QuantumEvolution(Qmodel = self.Qmodel, name="Observable",init=self.rho_0, obs=self.Obs)([noise_input, pulses])
            elif sim_type == "Observable_T":
                states  = QuantumEvolution(Qmodel = self.Qmodel, name="Observable",init=self.rho_0, obs=self.Obs)([noise_input, pulses])
                states  = layers.Lambda(lambda x: x[:,-1,:])(states)
            else:
                raise Exception("Wrong simulator type")   

            self.model    = Model([noise_input,pulses], states)
        
        self.__sim_type = sim_type
        self.model.summary() 
           
    @Qmodel.setter
    def Qmodel(self, Qmodel):
        self.__Qmodel = Qmodel
        self.noise    = self.Qmodel.noise
        self.__K = self.Qmodel.constants.num_realizations
        
    #@targets.setter
    #def targets(self, targets):
    #    if targets != None:
    #        self.Qmodel.rho_0   = rho_0(self.Qmodel.dim/self.Qmodel.dim_sys, targets["state"])
    #        self.Qmodel.Obs     = Obs(self.Qmodel.dim/self.Qmodel.dim_sys, targets["observable"])
    #        self.__targets = targets
    #    elif targets == None and self.Qmodel.dim != 2:
    #        raise Exception("targets are missing")
    
    #self.rho_0 = rho_0(self.dim/self.dim_sys, rho)
    #self.Obs = Obs(self.dim/self.dim_sys, O)
    
    @init_states.setter
    def init_states(self, init_states):
        if init_states is None:
            init_states = standard_init(self.Qmodel.dim_sys) 
        
        self.rho_0   = rho_0(self.Qmodel.dim/self.Qmodel.dim_sys, init_states)
        self.__init_states = init_states
        if self.rho_0[0].shape[0] != self.Qmodel.dim:
            raise Exception("Wrong dimension for initial states.")
      
    @observables.setter
    def observables(self, observables):
        if observables is None:
            observables = standard_obs(self.Qmodel.dim_sys)

        self.Obs     = Obs(self.Qmodel.dim/self.Qmodel.dim_sys, observables)
        self.__observables = observables
        if self.Obs[0].shape[0] != self.Qmodel.dim:
            raise Exception("Wrong dimension for observables.")
    
        
    ################################################################################   
    
    def save_Simulator(self, filename):
        with open(filename, 'wb') as file:
            pickle.dump(self, file)
    
    def load_Simulator(self, filename):
        with open(filename, 'rb') as file:
            simulator = pickle.load(file)
        
        return simulator

    
    ################################################################################   
     



    ################################################################################   


    def process_fidelity(self, G, pulses, batch_size=1):
        """
        sim_type = Choi, classical noise
        pulses should be (K,M,2)
        
        ## return Choi, process fidelity
        """
        pulses = np.tile(pulses, [self.__K,1,1])
        if self.sim_type == "Choi":
            EPR    = tf.constant( np.array([[1],[0],[0],[1]]) @ np.array([[1],[0],[0],[1]]).conj().T, dtype=tf.complex128) # unnormalized (why? quantum information theory) 
            choi_G = np.reshape( np.kron(np.eye(2), G) @ EPR @ np.kron(np.eye(2), G).conj().T, (1,4,4) ).astype(np.complex64)
            fid1 = process_fidelity2(0.5*choi_G[0,:], 0.5*tf.reduce_mean(self.model.predict([self.noise,pulses], batch_size = batch_size), 0))
            self.fidelity = fid1.numpy()
            return self.fidelity
        else:
            raise Exception("Simulator type should be Choi")
        

    def simulate_all(self, pulses, batch_size=1):
        #We might want to simulate with different pulses
        #pulses should be (K,M,2)
        
        pulses = np.tile(pulses, [self.__K,1,1])
     
        if self.sim_type in ["State_t", "State_T", "State_Observable_t", "State_Observable_T"]:
            #(1,6,K,M,dim,dim)
            return np.mean(self.model.predict([self.noise,pulses], batch_size = batch_size), axis=0)
        elif self.sim_type == "Observable_t" or self.sim_type == "Observable_T":
            #(K,M,n_obs)
            return np.mean(self.model.predict([self.noise,pulses], batch_size = batch_size), axis=0, keepdims=True)
        else:
            raise Exception("Wrong simulator type")
       
    def simulate(self, pulses, batch_size = 1, init=None, obs=None):
        sim = ["State_t","State_T","Observable_t","Observable_T","State_Observable_t","State_Observable_T"]
        self.data  = []
        if init is not None:
            self.rho_0 = init

        if obs is not None:
            self.Obs = obs

        if self.sim_type == "Instantaneous":
            expectations = self.model.predict(pulses, batch_size=batch_size) 
            self.data.append(expectations)
        elif self.sim_type in sim:
            states = self.simulate_all(pulses, batch_size)
            
            if self.Qmodel.noise_parameters is not None:
                if self.Qmodel.noise_parameters.noise_type == "Quantum":
                    if "State" in self.sim_type:
                        n_tot = self.Qmodel.num_qubits + self.Qmodel.num_aux
                        start = self.Qmodel.num_qubits +1
                        states = ptrace(states, n_tot, start,n_tot)
                        self.Obs = self.observables
  
            self.data.append(states) #(1,n_init,M,dim,dim) or (1,n_obs,M)
            #return self.data
            
            if self.sim_type == "State_Observable_t" or self.sim_type == "State_Observable_T":
                self.data.append(self.expectations())
        else:
            raise Exception("Wrong simulator type.")            
            
    def expectations(self):
        n_obs = len(self.Obs)
        n_exp = len(self.rho_0)*n_obs
        if "_t" in self.sim_type:
            obs = [np.real(np.trace(self.data[0][:,i//n_obs:(i//n_obs)+1,:,:,:] @self.Obs[i%n_obs], axis1=3, axis2=4)) for i in range(n_exp)]
            obs = np.concatenate(obs, axis = 1)
        elif "_T" in self.sim_type:
            obs = [np.real(np.trace(self.data[0][:,i//n_obs:(i//n_obs)+1,:,:] @self.Obs[i%n_obs], axis1=2, axis2=3)) for i in range(n_exp)]
            obs = np.concatenate(obs, axis = 1)     
    
        return obs
    
    def get_rho_t(self):
        if self.sim_type in ["State_t", "State_Observable_t"]:
            return self.data[0]
        else:
            raise Exception("Wrong simulation type.")
    
    def get_obs_t(self):
        if self.sim_type in ["Observable_t", "State_Observable_t"]:
            return self.data[-1]
        elif self.sim_type =="State_t":
            return self.expectations()
        else:
            raise Exception("Wrong simulator type.")
            
    def get_rho_T(self):
        if self.sim_type in ["State_t", "State_Observable_t"]:
            return self.data[0][:,:,-1,:,:]
        elif self.sim_type in ["State_T","State_Observable_T"]:
            return self.data[0]
        else:
            raise Exception("Wrong simulator type.")
            
    def get_obs_T(self):
        if self.sim_type in ["Observable_t","State_Observable_t"]:
            return self.data[-1][:,-1,:]
        elif self.sim_type in ["Observable_T", "State_Observable_T","Instantaneous"]:
            return self.data[-1]
        elif self.sim_type == "State_t":
            return self.expectations()[:,-1,:]
        elif self.sim_type == "State_T":
            return self.expectations()
        else:
            raise Exception("Wrong simulator type.")  
        

    def get_chi_T(self, pulses):
        #actual
        chi_dim = (self.Qmodel.dim)**2
    
        sigma   = sigma_dim(self.Qmodel.num_qubits)
        ##### sigma -> rho_0 and Obs
        A       = np.zeros((chi_dim**2,chi_dim**2), dtype=np.complex64)
        b       = np.zeros((chi_dim**2,1))
        chi     = np.zeros((chi_dim,chi_dim), dtype=np.complex64)
        print("A", A.shape)
        idx_row = 0
        self.simulate(pulses=pulses)

        for rho0, obs in product(self.rho_0,self.Obs):
            idx_col = 0
            for u in range(chi_dim):
                for v in range(chi_dim):
                    A[idx_row,idx_col] = np.trace(sigma[u]@rho0@sigma[v]@obs)
                    idx_col = idx_col + 1
            
            b[idx_row,0] = self.get_obs_T()[0,idx_row]
            idx_row = idx_row + 1

        sol     = np.linalg.inv(A)@b
        idx_row = 0
        for u in range(chi_dim):
            for v in range(chi_dim):
                chi[u,v] = sol[idx_row,0]
                idx_row = idx_row + 1

        self.chi = chi
        return chi
    
    def get_chi_gate(self, G):
        chi_dim = self.Qmodel.dim**2
        sigma = sigma_dim(self.Qmodel.num_qubits)
        simulate = lambda rho0, obs, G: np.real(np.trace(G@rho0@G.conj().T@obs))

        A       = np.zeros((chi_dim**2,chi_dim**2), dtype=np.complex64)
        b       = np.zeros((chi_dim**2,1))
        chi     = np.zeros((chi_dim,chi_dim), dtype=np.complex64)

        idx_row = 0
        for rho0, obs in product(self.rho_0,self.Obs):
            idx_col = 0
            for u in range(chi_dim):
                for v in range(chi_dim):
                    A[idx_row,idx_col] = np.trace(sigma[u]@rho0@sigma[v]@obs)
                    idx_col = idx_col + 1
            b[idx_row,0] = simulate(rho0,obs, G)
            idx_row = idx_row + 1

        sol     = np.linalg.inv(A)@b
        idx_row = 0
        for u in range(chi_dim):
            for v in range(chi_dim):
                chi[u,v] = sol[idx_row,0]
                idx_row = idx_row + 1

        return chi

    def get_chi_T_old(self, pulses):
        #actual
        #chi_dim = 4
        sigma   = [I,X,Y,Z]
        A       = np.zeros((16,16), dtype=np.complex64)
        b       = np.zeros((16,1))
        chi     = np.zeros((4,4), dtype=np.complex64)

        idx_row = 0
        self.simulate(pulses=pulses)

        for rho0, obs in product(sigma,sigma):
            idx_col = 0
            for u in range(4):
                for v in range(4):
                    A[idx_row,idx_col] = np.trace(sigma[u]@rho0@sigma[v]@obs)
                    idx_col = idx_col + 1
            
            b[idx_row,0] = self.get_obs_T()[0,idx_row]
            idx_row = idx_row + 1

        sol     = np.linalg.inv(A)@b
        idx_row = 0
        for u in range(4):
            for v in range(4):
                chi[u,v] = sol[idx_row,0]
                idx_row = idx_row + 1

        self.chi = chi
        return chi
    
    def get_chi_gate_old(self, G):
        sigma = [I,X,Y,Z]
        simulate = lambda rho0, obs, G: np.real(np.trace(G@rho0@G.conj().T@obs))

        A       = np.zeros((16,16), dtype=np.complex64)
        b       = np.zeros((16,1))
        chi     = np.zeros((4,4), dtype=np.complex64)

        idx_row = 0
        for rho0, obs in product(sigma,sigma):
            idx_col = 0
            for u in range(4):
                for v in range(4):
                    A[idx_row,idx_col] = np.trace(sigma[u]@rho0@sigma[v]@obs)
                    idx_col = idx_col + 1
            b[idx_row,0] = simulate(rho0,obs, G)
            idx_row = idx_row + 1

        sol     = np.linalg.inv(A)@b
        idx_row = 0
        for u in range(4):
            for v in range(4):
                chi[u,v] = sol[idx_row,0]
                idx_row = idx_row + 1

        return chi
    
    def chi_fidelity(self, chi_1, chi_2):
        return np.real(np.trace(chi_1.conj().T@chi_2))
    
def create_dataset(self, pulses,n_batch, filename, num_training, num_testing):
    #pulses shape should be [#dataset, M, dim]
    #store Qmodel, pulses, init_states, observable, state/exp depends on type
    data_y = np.zeros([pulses.params.N, 18])
    for i in range(pulses.params.N):
        self.simulate(pulses = pulses.pulses[i:i+1,:,:], batch_size = n_batch)
        if self.sim_type == "State_t":
            expec = self.get_rho_t()
        elif self.sim_type == "State_T":  
            expec = self.get_rho_T()
        elif self.sim_type == "Observable_t":  
            expec = self.get_obs_t()
        elif self.sim_type == "Observable_T":  
            expec = self.get_obs_T()
        data_y[i] = expec  
        
    training_x = [pulses.pulses_params[0:num_training,:,:], pulses.pulses[0:num_training, :, :]]
    training_y = data_y[0:num_training, :]
    testing_x = [pulses.pulses_params[num_training:num_training+num_testing, :, :], pulses.pulses[num_training+num_testing, :, :]] 
    testing_y = data_y[num_training+num_testing, :]
    np.savez(filename, training_x1 = training_x[0], training_x2 = training_x[1], training_y = training_y, testing_x1 = testing_x[0], testing_x2 = testing_x[1], testing_y = testing_y)