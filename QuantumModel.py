from constants import *

class QuantumModel():
    def __init__(self, operators=None, constants=None, noise_parameters=None, num_qubits =1, num_aux = 0, time_range = []):
        """
        Class constructor
        
        operators        : {drift, control, noise, collapse, measurement}
            drift        : drift
            control      : control
            noise        : noise (arbitrary dimensions)
            collapse     :
            measurement  : the O's, default [X,Y,Z]
        constants        : dictionary of constants for system+aux {drift, collapse, T, num_steps, num_realizations, N, n_max, g, pulse_position}
            - pulse position should be [0,1] and unique, sort it in the construction
            - num_realizations = 1 (default)

        noise_parameters : {type, constants {gamma, sd}}
        num_qubits         : number of qubits in the system
        """ 
        
        """
        Attributes:
        
        operators        : dictionary of operators
        constants        : dictionary of constants
        noise_parameters : dictionary of noise_parameters
        time_range       : time range
        dim              : dimension of system + auxiliary (aux=0 if noise is classical)
        dim_sys          : dimension of system
        noise            : noise generated according to noise parameters
        V_O              : basis of V_O
        """
        #if user specifies collapse, pulse position then it's a new simulation
        
        self.operators        = operators
        self.constants        = constants
        self.time_range       = time_range 
        self.noise_parameters = noise_parameters
        self.num_aux          = num_aux
        self.num_qubits       = num_qubits 
                     
           
    ################################################################################
     #Getters
    @property
    def operators(self):
        return self.__operators
    
    @property
    def constants(self):
        return self.__constants
    
    @property
    def noise_parameters(self):
        return self.__noise_parameters
    
    @property
    def time_range(self):
        return self.__time_range

    @property
    def num_aux(self):
        return self.__num_aux
      
    @property
    def num_qubits(self):
        return self.__num_qubits
    
    
    ################################################################################     
    #Setters
    @operators.setter
    def operators(self, operators):
        if operators is None:
            self.__operators = None
        else:
            operators.setdefault("measurement", [X,Y,Z])
            self.__operators = objdict(operators)
        
    @constants.setter
    def constants(self, constants):
        if constants is None:
            self.__constants = None
        else:
            constants.setdefault("num_realizations", 1)
            self.__constants = objdict(constants)
            self.__T = self.constants.T
            self.__M = self.constants.num_steps
            self.__K = self.constants.num_realizations
            
            if "g" in constants.keys():
                 # cast g to list of float
                if len(self.constants.g) == len(self.operators.noise):
                    for idx in range(len(self.constants.g)):
                        self.constants.g[idx] = float(self.constants.g[idx])
                else:
                    raise Exception("g has incorrect dimension.")
                
            if "pulse_position" in constants.keys():
                # sort the pulse position
                constants["pulse_position"].sort()
                self.__pulse_position = constants["pulse_position"]
                self.num_pulses = len(self.__pulse_position)

            if len(self.constants.drift) != len(self.operators.drift):
                # drift constant's dimension should be the same as operator's dimension
                raise Exception("drift has incorrect dimension.")
        
    @noise_parameters.setter
    def noise_parameters(self, noise_parameters):
        if noise_parameters is None:
            self.__noise_parameters = None
        else:
            noise = ["RTN", "RTN_mod", "Quasi_static", "Gaussian", "Quantum", "SC_noise"]
    
            if noise_parameters["noise_type"] in noise:
                if noise_parameters["noise_type"] in ["RTN", "RTN_mod", "Gaussian", "SC_noise"]:
                    if not("constants" in noise_parameters.keys()):
                        raise Exception("constants is not defined")
                    
                self.__noise_parameters = objdict(noise_parameters)
                if self.noise_parameters.noise_type == "Quantum":
                    self.constants.update({"num_realizations":1})
                elif self.noise_parameters.noise_type in ["RTN", "RTN_mod"]:
                    if "gamma" in self.noise_parameters.constants.keys():
                        if len(self.noise_parameters.constants.gamma) != len(self.operators.noise):
                            raise Exception("gamma has incorrect dimension.")
                    else:
                        raise Exception("gamma is not defined")
                     
                    if self.noise_parameters.noise_type == "RTN_mod":
                        if not("Omega_mod" in self.noise_parameters.constants.keys()):
                            raise Exception("Omega_mod is not defined.")
                        if len(self.noise_parameters.constants.Omega_mod) != len(self.operators.noise):
                            raise Exception("Omega_mod has incorrect dimension.")
                elif self.noise_parameters.noise_type == "Gaussian":
                    if not("sd" in self.noise_parameters.constants.keys()):
                        raise Exception("sd is not defined.")
                    if len(self.noise_parameters.constants.sd) != len(self.operators.noise):
                        raise Exception("sd has incorrect dimension.")
                elif self.noise_parameters.noise_type == "SC_noise":
                    
                    if not("alpha" in self.noise_parameters.constants.keys()):
                        raise Exception("alpha is not defined.")
    
            else:
                raise Exception("Wrong noise type.")
            
            self.noise = self.generate_noise()       
    
    @time_range.setter
    def time_range(self,time_range):
        if time_range == [] and self.constants is not None:
            self.__time_range = np.array([(0.5*self.__T/self.__M) + (j*self.__T/self.__M) for j in range(self.__M)]) 
        else:
            self.__time_range = time_range
    
    @num_aux.setter
    def num_aux(self,num_aux):
        self.__num_aux = num_aux
        if self.noise_parameters is not None:
            if self.noise_parameters.noise_type == "Quantum" and num_aux==0:
                self.__num_aux = 1
                

    @num_qubits.setter
    def num_qubits(self, num_qubits):
        if self.operators is not None:
            d = self.operators.drift[0].shape[0]
            self.__num_qubits = num_qubits
            self.dim_sys = 2**self.num_qubits  #system dimension
            self.dim = self.dim_sys*(2**(self.num_aux))

            if self.dim != d:
                raise Exception("Number of qubits is incorrect.")
    

                
            
    ################################################################################  
    #Hamiltonian
    def H_d(self):
        return sum([a*b for a,b in zip(self.operators.drift, self.constants.drift)])
        
    def H_control(self,pulses):
        #pulses is a list of objects
        return sum([a*b for a,b in zip(self.operators.control, pulses)])
        
    def H_1(self, noise):
        #noise is a list of objects, make it all 1s for quantum
        return sum([a*b for a,b in zip(self.operators.noise, noise)])        
     
    ################################################################################  
    #Save and Load
    
    def save_Qmodel(self, filename):
        with open(filename+'.qm', 'wb') as file:
            pickle.dump(self, file)
    
    def load_Qmodel(self, filename):
        with open(filename+'.qm', 'rb') as file:
            Qmodel = pickle.load(file)
        
        return Qmodel
   
    ################################################################################
    #Generate noise
    #Noise type: RTN, RTN mod, Quasi static, Gaussian, Quantum
    def generate_noise(self):  
        #noise shape : (K,M,#noise operators)
        axes = len(self.operators.noise)
        noise = ["RTN", "RTN_mod", "Quasi_static", "Gaussian", "Quantum", "SC_noise"]

        if (self.noise_parameters.noise_type in noise):
            if (self.noise_parameters.noise_type == "RTN"):
                self.noise = [self.constants.g[idx]*self.generate_RTN_noise(self.noise_parameters.constants.gamma[idx]) for idx in range(axes)]
            elif (self.noise_parameters.noise_type == "RTN_mod"):
                self.noise = [self.constants.g[idx]*self.generate_RTN_mod(self.noise_parameters.constants.gamma[idx], self.noise_parameters.constants.Omega_mod[idx]) for idx in range(axes)]            
            elif (self.noise_parameters.noise_type == "Quasi_static"):
                self.noise = [self.constants.g[idx]*self.generate_Quasi_static() for idx in range(axes)] 
            elif (self.noise_parameters.noise_type == "Gaussian"):
                self.noise = [self.constants.g[idx]*self.generate_Gaussian(self.noise_parameters.constants.sd[idx]) for idx in range(axes)]
            elif (self.noise_parameters.noise_type == "Quantum"): 
                self.noise = [self.constants.g[idx]*np.ones([1, self.__M,1], dtype = np.complex128) for idx in range(axes)]
            elif (self.noise_parameters.noise_type == "SC_noise"): #alpha's are part of noise_parameters.constants
                f  = np.fft.fftfreq(self.__M)*self.__M/self.__T 
                alpha1 = self.noise_parameters.constants.alpha[0]
                alpha2 = self.noise_parameters.constants.alpha[1]
                S_Z = np.array([(alpha1/fq) + alpha2*fq for fq in f[f>=0]]); S_Z[0] = S_Z[1]
                self.P_desired = S_Z
                self.noise = [self.constants.g[idx]*self.generate_arbitrary_noise()[0] for idx in range(axes)] #to make the shape consistent
                
        else:
            raise Exception("Wrong noise!")
    
        self.noise = np.concatenate(self.noise, axis = -1)  
        
        return self.noise
    
    def generate_RTN(self, gamma):
        return (2*(np.random.rand()>0.5)-1)* ((-1)**(np.cumsum(np.random.poisson(gamma*self.__T/self.__M, self.__M))))

    def generate_RTN_noise(self, gamma):
        return np.concatenate([np.reshape(self.generate_RTN(gamma) , (1,self.__M,1) ) for _ in range(self.__K)], axis=0)
    
    def generate_RTN_mod(self, gamma, omega):
        return np.concatenate( [np.concatenate( [np.reshape( np.cos(omega*self.time_range + 2*np.pi*np.random.rand())*self.generate_RTN(gamma) , (1,self.__M,1) ) for _ in range(self.__K)], axis=0) for _ in range(1)], axis=0) 

    def generate_Gaussian(self,sd):
        return np.random.normal(0, sd, (self.__K,self.__M,1))
    
    def generate_Quasi_static(self):
        noise = (2*np.random.rand(self.__K,1,1)-1) 
        return np.tile(noise, (1,self.__M,1))



    def generate_arbitrary_noise(self):
            """
            generate random noise according to some desired power spectral density according to the algorithm here:
    https://stackoverflow.com/questions/25787040/synthesize-psd-in-matlab
            P_desired: an array representing the desired PSD [single side band representation]
            """
            Ts = self.__T/self.__M  # sampling time (1/sampling frequency)
            N  = self.__M    # number of required samples
            # define a list to store the different noise realizations
            beta  = []
    
            # generate different realizations
            for _ in range(self.__K):
                #1) add random phase to the properly normalized PSD
                P_temp = np.sqrt(self.P_desired*N/Ts)*np.exp(2*np.pi*1j*np.random.rand(1,N//2))
    
                #2) add the symmetric part of the spectrum
                P_temp = np.concatenate( ( P_temp , np.flip(P_temp.conj()) ), axis=1 )
    
                #3) take the inverse Fourier transform
                x      = np.real(np.fft.ifft(P_temp))
    
                # store
                beta.append(np.reshape(x,(1,1,self.__M,1)))
            beta = np.concatenate(beta, axis=1)
            return beta #shape: (1,K,M,1)    


    ################################################################################  
    # Generate VO
    
    def generate_V_O(self, batch_size):  
        sigma = [I,X,Y,Z]
        dim_v = [self.__K//batch_size]+[4 for _ in range(2*self.num_pulses)] + [2,2]
        V = [np.zeros(dim_v, dtype = np.complex128) for O in self.operators.measurement]
        H_d = tf.cast(self.H_d(), dtype = tf.complex128) #(4,4)
        delta_t = self.__T/self.__M 
        
        for batch_number in range(self.__K//batch_size):     
            noise = [tf.tile(tf.expand_dims(tf.cast(self.noise[batch_number*batch_size:(batch_number+1)*batch_size,:,idx_n:idx_n+1], dtype = tf.complex128), axis=-1), [1,1,self.dim,self.dim]) for idx_n in range(self.noise.shape[-1])]
            H_1 = self.H_1(noise)
            H = H_d + H_1 #(K,M,2,2)
   
            #Unitary

            U = tf.linalg.expm(-1j*H*delta_t) #each element is e^{-iH(delta_t)delta_t}, shape same as H  
            
            U_list = []

            for idx_u in range(self.num_pulses + 1):
                if idx_u == 0:
                    start = 0
                    end = int(self.__pulse_position[idx_u]*self.__M) -1
                elif idx_u == self.num_pulses:
                    start = int(self.__pulse_position[idx_u-1]*self.__M)
                    end = self.__M-1
                else:
                    start = int(self.__pulse_position[idx_u-1]*self.__M)
                    end = int(self.__pulse_position[idx_u]*self.__M) -1
                
                U_list.append(reduce(lambda x, y: tf.matmul(y, x), [ U[:, idx_t, :,  : ] for idx_t in range(start,end+1)]))
                   #(),(),()
            
            for alpha_string in product(range(4),repeat=self.num_pulses*2): #[0,0,0,0,:,:] [0,1,0,0,0,1] 2n-1, 2n-2, 2n-n...
                U_copy = U_list.copy()
                U_dag_copy = U_list.copy()
                #[0,1,2]=> on 1 => [0,x,1,2] =>on 3 => 0,x,1,x,2
                for idx_pos in range(self.num_pulses):
                    U_copy.insert(2*idx_pos+1, sigma[alpha_string[idx_pos]])    
                    U_dag_copy.insert(2*idx_pos+1, sigma[alpha_string[self.num_pulses +idx_pos]]) #Ua_1'Ua_2'U
                
                U_copy = reduce(lambda x, y: tf.matmul(y, x), U_copy)
                U_dag_copy = tf.transpose(reduce(lambda x, y: tf.matmul(y, x), U_dag_copy), perm = (0,2,1) , conjugate = True)
        
                index = (slice(batch_number,batch_number+1),) + tuple([slice(idx,idx+1) for idx in alpha_string]) + (slice(0,2),)*2
                #index = tuple([batch_number]) + tuple(alpha_string) + (slice(0,2),)*2
                for idx_O,O in enumerate([X,Y,Z]):
                    V[idx_O][index] = tf.reduce_mean(tf.matmul(O, tf.matmul(U_dag_copy, tf.matmul(O,U_copy))), axis=0).numpy() #[1,4,4,4,4...2*num_pulses, 2, 2]
                    
        self.V_O = [np.mean(V[idx_O], axis=0, keepdims = True) for idx_O in range(3)]
       