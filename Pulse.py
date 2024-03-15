from constants import *
import warnings
warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning) 

class Pulse():
    def __init__(self, name=None, params=None, T=None, num_steps=None, time_range=[]):
        #should we be able to generate N pulses for all types?
        self.name       = name
        self.T          = T
        self.M          = num_steps
        self.time_range = time_range
        self.params     = params
    
    #Write a method to load pulses into the appropriate pulse type

    @property
    def time_range(self):
        return self.__time_range
    
    @property
    def name(self):
        return self.__name
    
    @property
    def params(self):
        return self.__params
    
    @time_range.setter
    def time_range(self,time_range):
        if time_range == []:
            self.__time_range = np.array([(0.5*self.T/self.M) + (j*self.T/self.M) for j in range(self.M)]) 
        else:
            self.__time_range = time_range
    
    @name.setter
    def name(self,name):
        pulse_names = ["Instantaneous", "Gaussian", "CPMG", "CPMG_XY", "Zero"]
        if name in pulse_names:
            self.__name = name
        elif name is None:
            pass
        else:
            raise Exception("Incorrect name.")
        
    @params.setter
    def params(self,params):
        if self.name != "Zero":
            if not("num_pulses" in params.keys()):
                raise Exception("Specify num_pulses.")
        
        if self.name == "Instantaneous":
            if not("position" in params.keys()):
                raise Exception("Specify position.")
            
        self.pulse_width = 0.5*self.T/params["num_pulses"]
        
        delimiter = ', '
        if self.name == "Instantaneous":
            p = ["num_pulses","A_max", "theta", "vec", "axes","position"]
            params.setdefault("A_max",2)
            params.setdefault("theta")
            params.setdefault("vec")
            params.setdefault("axes",3) 
            #if one of p not found in the params set it
            if not(all([key in p for key in params.keys()])):
                raise Exception("params should be any of (" + delimiter.join(p) +")")
        elif self.name == "Gaussian":
            p = ["num_pulses","N","amp_scale", "amplitude", "position","sd","axes"]
            params.setdefault("N",1)
            params.setdefault("amp_scale",1)
            params.setdefault("amplitude")
            params.setdefault("position")
            params.setdefault("sd", self.pulse_width/6)
            params.setdefault("axes", 1)
            if not(all([key in p for key in params.keys()])):
                raise Exception("params should be any of ("+  delimiter.join(p) +")")
        elif self.name == "CPMG":
            p = ["num_pulses","A_max"]
            params.setdefault("A_max",1)
            if not(all([key in p for key in params.keys()])):
                raise Exception("params should be any of ("+  delimiter.join(p) +")")
            params.setdefault("axes",1)
        elif self.name == "CPMG_XY":       
            params.setdefault("axes",2)
        elif self.name == "Zero":
            p = ["axes"]
            params.setdefault("axes", 3)
            if not(all([key in p for key in params.keys()])):
                raise Exception("params should be any of ("+  delimiter.join(p) +")")

        self.__params = objdict(params)
    
    
    #-----------------------------------------------------------------------------
    # Save and Load (extension .pls, no need to put extension when saving/loading)
    #-----------------------------------------------------------------------------
    
    def save_Pulse(self, filename):
        with open(filename+".pls", 'wb') as file:
            pickle.dump(self,file)
    
    def load_Pulse(self, filename):
        with open(filename+".pls", 'rb') as file:
            pulse = pickle.load(file)
        
        return pulse
    
    #--------------------------------------------------------------------------
    # Generate Pulse
    #--------------------------------------------------------------------------  
    
    def generate_pulse(self):

        if self.name == "Instantaneous":
            self.generate_Instantaneous_pulse(num_pulses = self.params.num_pulses, A_max=self.params.A_max, theta=self.params.theta, vec=self.params.vec,axes=self.params.axes)
        elif self.name == "Gaussian":
            self.generate_N_pulses(num_pulses = self.params.num_pulses, N=self.params.N, amp_scale=self.params.amp_scale, amplitude=self.params.amplitude, position=self.params.position, sd=self.params.sd, axes=self.params.axes)
        elif self.name == "CPMG":
            self.generate_CPMG(num_pulses = self.params.num_pulses,A_max=self.params.A_max)
        elif self.name == "CPMG_XY":
            self.generate_CPMG_XY(num_pulses = self.params.num_pulses)
        elif self.name == "Zero":
            self.pulses = np.zeros[1,self.M, self.params.axes]
            self.pulses_params = []
            
    
    #--------------------------------------------------------------------------
    def generate_C_1(self,num_pulses,A_max = 2, theta=None,vec=None):
        C_arr = 1
        params = []
        for idx in range(num_pulses):
            c_row = np.zeros([4], dtype = np.complex128)
            if theta ==None:
                Theta = np.random.uniform(0,np.pi) *A_max
            else:
                Theta = theta[idx]
                
            if vec == None:
                n = np.random.rand(3)
            else:
                n = vec[idx]
            
            n = n/np.linalg.norm(n)
            
            
            c_row[0] = np.cos(Theta)
            c_row[1] = -1j*np.sin(Theta)*n[0]
            c_row[2] = -1j*np.sin(Theta)*n[1]
            c_row[3] = -1j*np.sin(Theta)*n[2]
            
            C_arr = np.tensordot(C_arr, c_row, axes = 0)
            
            params = np.append(params, [Theta,n[0],n[1],n[2]])
            
        return C_arr, params
    
    def generate_C(self,num_pulses, A_max=2,theta=None,vec=None,axes=3):
        #vec = list of list
        #theta = list
        if num_pulses == 0:
             return 1,[]
        else:
            c_row = np.zeros([4], dtype = np.complex128)
            if theta is None:
                Theta = np.random.uniform(0,np.pi) *A_max
            else:
                Theta = theta[num_pulses-1]
         
            if vec is None:
                n = np.concatenate((np.random.rand(axes), np.array([0 for _ in range(3-axes)])))
            else:
                n = vec[num_pulses-1]
            
            n = n/np.linalg.norm(n)
            
            c_row[0] = np.cos(Theta)
            c_row[1] = -1j*np.sin(Theta)*n[0]
            c_row[2] = -1j*np.sin(Theta)*n[1]
            c_row[3] = -1j*np.sin(Theta)*n[2]    
            
            params = [Theta,n[0],n[1],n[2]]
            
            C = self.generate_C(num_pulses-1, A_max, theta, vec, axes)
       
            C_pulse = np.tensordot(C[0],c_row, axes = 0)
            C_params = np.append(C[1],params)   
            
            return C_pulse, C_params
        
    def generate_Instantaneous_pulse(self, num_pulses=None, A_max=2, theta=None, vec=None,axes=3):
        if num_pulses == None:
            num_pulses = self.params.num_pulses
        else:
            num_pulses = num_pulses
            
        C = self.generate_C(num_pulses,A_max,theta,vec,axes)
        
        self.pulses = np.expand_dims(np.tensordot(C[0], C[0].conj(), axes=0), axis=0)
        self.pulses_params = np.reshape(C[1], (num_pulses, 4))
        return self.pulses, self.pulses_params 
    
    def insta_to_gaussian_Amp(self,sd):
        #axes = 0,1,2
        insta_params = self.pulses_params
        #num_pulses = insta_params.shape[0]
            
        amplitude = [insta_params[idx,0]*insta_params[idx,axes+1]/(np.sqrt(np.pi)*sd) for axes,idx in product(range(self.params.axes), range(self.params.num_pulses))]
        return np.reshape(amplitude, [self.params.axes,self.params.num_pulses])
    
    def convert_to_gaussian(self,position=None,sd=None):
        #returns an object pulse with name=Gaussian
        # Gaussian position is in seconds, conver insta position by multiplying by T
        if position ==None:
            position = np.array(self.params.position) 
        else:
            position = np.array(position)
            
        if sd == None:
            sd= self.pulse_width/6
        else:
            sd = sd
            
        if self.name=="Instantaneous":
            amplitude = self.insta_to_gaussian_Amp(sd)
            params = {"num_pulses": self.params.num_pulses,"amplitude":amplitude, "position":position, "sd":sd, "axes":len(amplitude)}
            pulse_gaussian = Pulse(name="Gaussian", params=params, T=self.T, num_steps=self.M)
            pulse_gaussian.generate_pulse()
            return pulse_gaussian
        else:
            raise Exception("Wrong pulse type.")
        
    
    def generate_Gaussian_pulse(self, num_pulses=None,amp_scale=1, amplitude= None, position=None,sd=None):
        # Define a function that generate a random Gaussian control sequence, such that no to pulse overlap. Randomization is over
        # the location of the pulse, and its amplitude. [one examples]
        # change = True: user-defined A_max is used
        # A_max is either None, int/float, or list
        # position is either None or list
        ######################################################## 
        # arrays needed for generating the control pulses
        ######################################################### 
        
        if num_pulses == None:
            num_pulses = self.params.num_pulses
        else:
            num_pulses = num_pulses
        
        A_max = amp_scale* self.pi_pulse(self.params.sd )
      
        if amplitude is None:               
            amplitude = np.random.uniform(low=-A_max, high=A_max, size = (num_pulses,1)) 
        else:
            amplitude = np.reshape(amplitude,[num_pulses,1])
        
        if position is None:
            a_matrix    = np.ones((num_pulses, num_pulses))
            a_matrix[np.triu_indices(num_pulses,1)] = 0
            b_matrix    = np.reshape([idx + 0.5 for idx in range(num_pulses)], (num_pulses,1) ) * self.pulse_width

            position    = 0.5*self.pulse_width + np.random.uniform(size=(num_pulses,1))*( ( (self.T - num_pulses*self.pulse_width)/(num_pulses+1) ) - 0.5*self.pulse_width)
            position    = a_matrix @ position + b_matrix
               
        else:
            # position was normalized, should be rescaled by T
            position = np.reshape(position,(self.params.num_pulses,1))*self.T
        
        waveform    =  np.sum([A*np.exp(-((self.time_range-tau)/self.params.sd)**2) for A,tau in zip(amplitude,position)], axis=0)
        
        return 0.5*(1 + (amplitude/A_max) ), position/self.T, waveform

    def generate_N_pulses(self, num_pulses=None, N=1, amp_scale=1, amplitude=None, position=None, sd=None, axes=1):  
        # the full dataset
        # Initialize arrays for storing the dataset formatted for tensorflow applications
        # A_max = list of list or int or float, len(A_max) = #axes
        # params = [[amplitude], [position]]
        if num_pulses == None:
            num_pulses = self.params.num_pulses
        else:
            num_pulses = num_pulses
        
        if sd == None:
            sd = self.pulse_width/6
        else:
            sd = sd
        
        self.params.sd = sd
            
        data_x1      = np.zeros((N,num_pulses,2*axes))
        data_x2      = np.zeros((N,self.M,axes))

        for idx_ex in range(N):
            for idx_a in range(axes):
                if amplitude is None:
                    amp, pos, waveform = self.generate_Gaussian_pulse(num_pulses=num_pulses, amp_scale=amp_scale, position=position)
                else:
                    amp, pos, waveform = self.generate_Gaussian_pulse(num_pulses=num_pulses, amp_scale=amp_scale, amplitude=amplitude[idx_a], position=position) # x-axis pulses

                data_x1[idx_ex,:,(2*idx_a):(2*idx_a+1)]  = amp            
                data_x1[idx_ex,:,(2*idx_a+1):(2*idx_a+2)]  = pos

                data_x2[idx_ex,:,idx_a]    = waveform

        self.pulses = data_x2
        self.pulses_params = data_x1
        return self.pulses, self.pulses_params

    def generate_CPMG(self,num_pulses=None,A_max=1):
        # Define a function that generate a random Gaussian control sequence, such that no to pulse overlap. Randomization is over
        # the location of the pulse, and its amplitude. [one examples]
        ######################################################## This can be moved to class constructor
        # arrays needed for generating the control pulses
        if num_pulses == None:
            num_pulses = self.params.num_pulses
        else:
            num_pulses = num_pulses
        
        sd       = 6*self.T/self.M   # pulse width
        position  = [self.T*(k-0.5)/num_pulses for k in range(1,num_pulses+1)] # position [equally space]
        amplitude = A_max*[self.pi_pulse(sd) for _ in range(1,num_pulses+1)]  # pi amplitude pulse 
        #print(position)
        #########################################################     
        waveform  = np.sum([A*np.exp(-((self.time_range-tau)/sd)**2) for A,tau in zip(amplitude,position)], axis=0)
        
        self.pulses = np.reshape(waveform, (1,self.M,1))
        #return self.waveform

    def generate_CPMG_XY(self,num_pulses=None):
        # Define a function that generate a random Gaussian control sequence, such that no to pulse overlap. Randomization is over
        # the location of the pulse, and its amplitude. [one examples]
        ######################################################## This can be moved to class constructor
        # arrays needed for generating the control pulses
        if num_pulses == None:
            num_pulses = self.params.num_pulses
        else:
            num_pulses = num_pulses
        
        sd         = 6*self.T/self.M # pulse width
        position_x  = [self.T*(k-0.5)/num_pulses for k in range(1,num_pulses+1,2)] # position [equally space]
        amplitude_x = [self.pi_pulse(sd)  for _ in range(1,num_pulses+1,2)] # pi amplitude pulse     
        position_y  = [self.T*(k-0.5)/num_pulses for k in range(2,num_pulses+1,2)] # position [equally space]
        amplitude_y = [self.pi_pulse(sd) for _ in range(1,num_pulses+1,2)] # pi amplitude pulse 
        #########################################################     
        waveform_x    = np.sum([A*np.exp(-((self.time_range-tau)/sd)**2) for A,tau in zip(amplitude_x,position_x)], axis=0)
        waveform_y    = np.sum([A*np.exp(-((self.time_range-tau)/sd)**2) for A,tau in zip(amplitude_y,position_y)], axis=0)
        
        self.pulses = np.concatenate( [np.reshape(waveform_x, (1,self.M,1)), np.reshape(waveform_y, (1,self.M,1))], axis=-1)
        #return self.waveform

    def pi_pulse(self,sd):
        return np.pi/( np.sqrt(np.pi)*sd)