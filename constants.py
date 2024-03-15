from itertools import product, accumulate
import pickle
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers,optimizers,Model,regularizers,initializers
import math
import random
from functools import reduce
from scipy.linalg import expm
from scipy import signal, linalg



###############################################################################

#The Paulis
X = np.array([[0,1],[1,0]], dtype=np.complex128)
Y = np.array([[0, -1j],[1j,0]], dtype=np.complex128)
Z = np.array([[1,0],[0,-1]], dtype=np.complex128)
I = np.array([[1,0],[0,1]], dtype=np.complex128)

#Pauli eigenvectors
Xp = 0.5*np.array([[1.,1.],[1.,1.]], dtype=np.complex128) #X+
Xm = 0.5*np.array([[1.,-1.],[-1.,1.]], dtype=np.complex128) #X-
Yp = 0.5*np.array([[1.,-1j],[1j,1.]], dtype=np.complex128) #Y+
Ym = 0.5*np.array([[1.,1j],[-1j,1.]], dtype=np.complex128) #Y-
Zp = np.array([[1.0,0.0],[0.0,0.0]], dtype = np.complex128) #Z+
Zm = np.array([[0.0,0.0],[0.0,1.0]], dtype = np.complex128) #Z-


#======================================= Qubit ============================================= 
#Initial states
rho = [Xp, Xm, Yp, Ym, Zp, Zm]
#Observables
O = [I, X, Y, Z]

zero = np.array([[1,0],[0,0]], dtype = np.complex128)

######################################################################################################################

#======================================= Gates ===================================================

class Gates():
    def __init__(self, Gate, Gate_names):
        self.Gate = Gate
        self.Gate_names = Gate_names


####################################################################################################

def str_to_Pauli(s):
    if s =="I":
        return I
    elif s=="X":
        return X
    elif s=="Y":
        return Y
    elif s=="Z":
        return Z

def pauli(sigma):
    result = 1
    for s in sigma:
        if isinstance(s, str):
            s = str_to_Pauli(s)

        result = np.kron(result, s)
    return result

def sigma_dim(dim):
    if dim == 1:
        return [I,X,Y,Z]
    else:
        return [np.kron(y,z) for y,z in product ([I,X,Y,Z], sigma_dim(dim-1))]

def standard_init(dim):
    if dim == 1:
        return rho
    else:
        return [np.kron(y,z) for y,z in product (rho, standard_init(dim-1))]
    
def st_obs(dim):
    if dim == 1:
        return O
    else:
        return  [np.kron(y,z) for y,z in product (O, st_obs(dim-1))]

def standard_obs(dim):
    return st_obs(dim)[1:]
    
def rho_0(dim, rho0):
    if int(math.log(dim,2)) == 0:
        return rho0
    else:
        return np.kron(rho_0(dim//2, rho0),zero)
    
def Obs(dim, obs):
    if int(math.log(dim,2)) == 0:
        return obs
    else:
        return np.kron(Obs(dim//2, obs),I)

def trace_norm(rho):  
    return tf.math.real( tf.linalg.trace( my_sqrtm( tf.matmul(rho, rho, adjoint_b=True)) ) )

def set_basis(n_qubit):
    if n_qubit==0:
        return 1
    elif n_qubit==1:
        return [np.array([[1],[0]]), np.array([[0],[1]])]
    else:
        return [np.kron(set_basis(n_qubit-1)[idx1], set_basis(1)[idx2]) for idx1,idx2 in product(range(2**(n_qubit-1)), range(2))]

def ptrace(state, n_qubit, dim1,dim2=None):
    if dim2==None:
        dim2 = dim1

    basis = set_basis(dim2-dim1+1)
    I1 = np.eye(2**(dim1-1))
    I2 = np.eye(2**(n_qubit-dim2))
    
    arr = [np.kron(I1, np.kron(basis[idx].conj().T, I2)) @ state @ np.kron(I1, np.kron(basis[idx], I2)) for idx in range(2**(dim2-dim1+1))]
    return np.array(arr).sum(axis=0)



# loss function
def process_fidelity2( J_target, J_actual):
    # first argument is the the true, and the second is the prediction
    J_t = tf.cast(J_target,tf.complex128)
    J_a = tf.cast(J_actual, tf.complex128)
    
    x = tf.matmul( my_sqrtm(J_t), my_sqrtm(J_a) )
    return trace_norm( tf.matmul(x,x) )

def my_sqrtm(J):
    # for states, eig(J) is real
    D, Q = tf.linalg.eigh(J)
    D = tf.cast( tf.linalg.diag(tf.sqrt(tf.nn.relu(tf.math.real(D)))), tf.complex128)
    s = tf.matmul(Q, tf.matmul(D,Q, adjoint_b=True))
    #print(s)
    return s

# loss function
def process_fidelity2( J_target, J_actual):
    # first argument is the the true, and the second is the prediction
    J_t = tf.cast(J_target,tf.complex128)
    J_a = tf.cast(J_actual, tf.complex128)
    return tf.math.real( tf.linalg.trace( my_sqrtm( tf.matmul( tf.matmul( my_sqrtm(J_t), J_a), my_sqrtm(J_t)) ) ) )
    
def proc_fidelity( J_target, J_actual):
    # first argument is the the true, and the second is the prediction
    J_t = tf.cast(J_target,tf.complex128)/2
    J_a = tf.cast(J_actual, tf.complex128)/2
    return -tf.math.real( tf.linalg.trace( my_sqrtm( tf.matmul( tf.matmul( my_sqrtm(J_t), J_a), my_sqrtm(J_t)) ) ) )

# loss function
def matrix_norm(J_target, J_actual):
    # first argument is the the true, and the second is the prediction
    return tf.linalg.norm(tf.cast(J_target,tf.complex128) - tf.cast(tf.reduce_mean(J_actual, axis=0), tf.complex128 ))

class objdict_1(dict):
    def __getattr__(self, name):
        if name in self:
            return self[name]
        else:
            raise AttributeError("No such attribute: " + name)

    def __setattr__(self, name, value):

        if name in self:
            self[name] = value
        else:
            raise AttributeError("No such attribute: " + name)

    def __delattr__(self, name):
        if name in self:
            del self[name]
        else:
            raise AttributeError("No such attribute: " + name)
            
class objdict(dict):
    def __init__(self, *args, **kwargs):
        def from_nested_dict(data):
            """ Construct nested objdict from nested dictionaries. """
            if not isinstance(data, dict):
                return data
            else:
                return objdict({key: from_nested_dict(data[key])
                                    for key in data})

        super(objdict, self).__init__(*args, **kwargs)
        self.__dict__ = self

        for key in self.keys():
            self[key] = from_nested_dict(self[key])
            
#TO BE MOVED LATER
#161024
            

def generate_arbitrary_noise(P_desired, T, M, K):
        """
        generate random noise according to some desired power spectral density according to the algorithm here:
https://stackoverflow.com/questions/25787040/synthesize-psd-in-matlab
        P_desired: an array representing the desired PSD [single side band representation]
        """
        Ts = T/M  # sampling time (1/sampling frequency)
        N  = M    # number of required samples
        # define a list to store the different noise realizations
        beta  = []
 
        # generate different realizations
        for _ in range(K):
            #1) add random phase to the properly normalized PSD
            P_temp = np.sqrt(P_desired*N/Ts)*np.exp(2*np.pi*1j*np.random.rand(1,N//2))
 
            #2) add the symmetric part of the spectrum
            P_temp = np.concatenate( ( P_temp , np.flip(P_temp.conj()) ), axis=1 )
 
            #3) take the inverse Fourier transform
            x      = np.real(np.fft.ifft(P_temp))
 
            # store
            beta.append(np.reshape(x,(1,1,M,1)))
        beta = np.concatenate(beta, axis=1)
        return beta #shape: (1,K,M,1)    


def Haar(N):
    Q,R = linalg.qr( (np.random.randn(N,N) + 1j*np.random.randn(N,N) ) / np.sqrt(2))
    U   = Q @ np.diag([R[i, i] / np.abs(R[i, i]) for i in range(N)])
    return U / ( linalg.det(U)**(1/N) )


#alpha1    = (1e9) 
#alpha2    = (1e-9)
 
#Ts       = T/M
#f        = np.fft.fftfreq(M)*M/T 
# the rule is f/fs = k/N ==> f = (k/N)*fs. k=0 to N-1. maximum f is almost k/N=0.5, so it is 1/2T. So to see the higher frequncy components we need to work on a very tiny time scale. 
#print('f_max=%.2f MHz'%(np.max(f)/(1e6)))
#S_Z      = np.array([(alpha1/fq) + alpha2*fq for fq in f[f>=0]]); S_Z[0] = S_Z[1] #inside class or pass from outside?

#Qmodel.generate_arbitrary_noise(S_Z)
#beta     = generate_arbitrary_noise(S_Z, T, M, K)
##print("beta range = %.2f MHz"%((np.max(beta)-np.min(beta))/(1e6)))
#PSD_est  = sum([signal.periodogram(beta[0,k,:,0], 1/Ts)[1] for k in range(100)])/100
#f_est    = signal.periodogram(beta[0,0,:,0], 1/Ts)[0]
#plt.figure(figsize=[10,6]) plt.loglog(f_e.
#plt.figure(figsize=[10,6])
#plt.loglog(f_est[1:]/(1e9), PSD_est[1:], color="r", label="Estimated PSD")
#plt.loglog([fq/(1e9) for fq in f[f>0]],  [(alpha1/fq) for fq in f[f>0]], label=r"$\frac{\alpha_1}{f}$")
#plt.loglog([fq/(1e9) for fq in f[f>0]],  [alpha2*fq for fq in f[f>0]],  label=r"$\alpha_2 f$")
#plt.loglog([fq/(1e9) for fq in f[f>0]],  [(alpha1/fq) + (alpha2*fq) for fq in f[f>0]], color="k" , label=r"$\frac{\alpha_1}{f} + \alpha_2 f $")
 
##plt.xlabel("f (GHz)")
#plt.ylabel("S(f) (W/Hz)")
#plt.legend(ncol=4)
#plt.grid()
