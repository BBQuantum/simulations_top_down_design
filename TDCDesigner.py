
from linear_circuit import LinearCircuit
import numpy as np
from scipy.stats import unitary_group
from scipy.linalg import dft
class TDCDesigner:
    """Top Down (linear) Circuit Designer"""
    def __init__(self,
                channel_size,
                logic_size,
                number_of_planes,
                verbose=0):
        
        if logic_size>channel_size:
            raise(Exception('logic_size cannot be greater than channel_size.'))
        

        self.N=channel_size
        self.d=logic_size
        self.K=number_of_planes

        self.xports=None
        self.yports=None
        self.verbose=verbose
        self.circuit=LinearCircuit(verbose-1)
        self.plane_idxs=[]
        self.dont_update_these_planes=[]
        self.printVerbose('[[GenUnivUnit]] Initilized a new General Unitary')
        
    def setTarget(self,T):
        """
        Sets Target Gate we are trying to make here
        accepts only square matrcies
        """
        T=np.matrix(T)
        #T=T/np.sqrt(np.trace(T.conj().T @ T)/self.N)
        h,w=T.shape
        if h!=w:
            raise(Exception('Rectangular Gates cannot be implemented yet'))
        elif h!=self.d:
            raise(Exception('Dimension of gate and Number of inputs(d) not same'))
        else:
            self.T=T
            self.printVerbose('[[GenUnivUnit:SetTarget]] Setting Target Gate to :'+str(self.T))
        
        
    def printVerbose(self,message):
        """
        For printing Verbose messages
        """
        if self.verbose>0:
            print(message)
    
    def phasePlane(self,phis):
        """
        Represents a phase plane(SLM plane) in our Linear circuit
        
        Creates a diagonal matrix of Size N with complex phase on diagonal
        If the number of phase elements recieved is less than N, it pads 0s to the phases to make an NxN matrix
        """
        phis=np.array(phis)
        if len(phis)<self.N:
            np.pad(phis, (0,N-len(self.phis)), 'constant')        
        return(np.diag(np.exp(1j*phis)))
    
    def newPhasePlane(self,initphases='ones'):
        """
        Iniitalize new phase plane with either ones(i.e an identity), or with random phases
        """
        if initphases == 'ones':
             return self.phasePlane(np.zeros(self.N))
        elif initphases=='rand':
            return self.phasePlane(np.random.rand(self.N))
        
        
    def randPlane(self):
        """
        Random Unitary Transfer Matrix
        """
        V=np.matrix(unitary_group.rvs(self.N))
        
        V=V/np.sqrt(np.trace(V.conj().T @ V)/self.N)
        return(V)
    
    def dftPlane(self):
        """
        DFT Transfer Matrix
        """
        V=np.matrix(dft(self.N))
        V=V/np.sqrt(np.trace(V.conj().T @ V)/self.N)
        return(V)
        
    def IPlane(self):
        """
        Identity Transfer Matrix
        """
        return(np.identity(self.N))
    
        
    def initCircuit(self,type='DFT'):
        """         
        Initilialize linear circuit putting transfer layers and phase layers alternatively
        
        """
        if type=='RandSame': # Refers to same Random unitary matrix used for every transfer
            L=self.randPlane()
            layer=[L for i in range(self.K-1)]
            
        elif type=='RandDiff': # Refers to different Random unitary matrix used for every transfer
            layer=[self.randPlane() for i in range(self.K-1)]
        elif type=='Identity':  # Refers to Identity matrix used for every transfer
            layer=[self.IPlane() for i in range(self.K-1)]
        
        else:   # Refers to DFT matrix used for every transfer
            layer=[self.dftPlane() for i in range(self.K-1)]
            
        for k in range(self.K-1):  #
            self.printVerbose('[[GenUnivUnit:initDFTCircuit]] Adding Layer:'+str(k+1))
            self.circuit.add(self.newPhasePlane('ones'))
            self.circuit.add(layer[k])
        self.printVerbose('[[GenUnivUnit:initDFTCircuit]] Adding Layer:'+str(k+2))
        self.circuit.add(self.newPhasePlane('ones'))   #K phase planes, K-1 V(m)s, and last V(m) is Identity 
        
        self.plane_idxs=np.arange(0,2*self.K,2)+1
    

        
    def getTransform(self):
        if self.xports is None:
            self.setXYports()
        self.xd=np.identity(self.d)  #  #input vectors
        self.xs=np.zeros((self.d,self.N))+0j 
        self.xs[:,self.xports]=self.xd
          
        U=np.matrix([ self.circuit.solve(x,dir=1, pos=self.circuit.getCirLen())[self.yports] for x in self.xs]).T
        return U#return U
    
    def getFidelity(self):
        U=self.getTransform()
        F=(np.abs(np.trace(U.conj().T @ self.T))**2)/( np.abs(np.trace(U.conj().T @ U))*np.abs(np.trace(self.T.conj().T @ self.T))  )
        return F
    def getSuccessProb(self):
        U=self.getTransform()
        S=np.abs(np.trace(U.T.conj()@U)/self.d)
        return S
        

    def updatePlaneWFM(self,k):
        if k in self.dont_update_these_planes:
            return False
        invecs=self.vecs[0,k]
        outvecs=self.vecs[1,k]
        in_power=np.sum( np.abs(invecs)**2, axis=1)  # Power in each input mode: num_planes x num_modes
        out_power=np.sum( np.abs(outvecs)**2, axis=1)  # Power in each output mode: num_planes x num_modes        
        delta_masks_modes= invecs*outvecs.conj()/np.sqrt(in_power*out_power)[:,np.newaxis]
        delta_mask=np.sum(delta_masks_modes,axis=0)
        self.circuit.updateLayer(np.diag(np.exp(-1j*(np.angle(delta_mask)))),self.plane_idxs[k])
        return True
                            

            
  

    def setXYports(self,type='topfixed'):
        """
        To set which input and output modes we select to make our target gate in this circuit. works only for d<N
        
        """
        if(type=='topfixed'):
            self.xports=np.array(np.pad(np.ones(self.d),(0,self.N-self.d),'constant'),dtype=bool)
            self.yports=np.array(np.pad(np.ones(self.d),(0,self.N-self.d),'constant'),dtype=bool)
        
            
        elif(type=='randxy'):
                        
            self.xports=np.array(np.random.permutation(np.pad(np.ones(self.d),(0,self.N-self.d),'constant')),dtype=bool)
            self.yports=np.array(np.random.permutation(np.pad(np.ones(self.d),(0,self.N-self.d),'constant')),dtype=bool)
        else:
            raise(Exception('Wrong XY port Type'))

    def WFM(self,niters=1000):
        F=self.getFidelity()
        dF=0.0001
        dS=0.0001
        patienceFid=100
        maxF=0
        maxS=0
        if self.xports is None:
            self.setXYports()
        self.xd=np.identity(self.d)  #  #input vectors
        self.yd=self.T.T #target ourputs
        
        self.xs=np.zeros((self.d,self.N),dtype=np.complex)
        self.ys=np.zeros((self.d,self.N),dtype=np.complex)
    
        self.xs[:,self.xports]=self.xd
        self.ys[:,self.yports]=self.yd
        
        self.vecs=np.zeros((2,self.K,*self.xs.shape),dtype=np.complex )
  
        for k in range(self.K):
            self.vecs[0,k]=self.circuit.solve(self.xs,dir=1,pos=self.plane_idxs[k]-1) 
            self.vecs[1,k]=self.circuit.solve(self.ys,dir=-1,pos=self.plane_idxs[k]) 

        
        stopCount=0
        i=0
        for i in range(niters):
            self.updatePlaneWFM(0)
            for i in np.arange(1,self.K):
                self.vecs[0,i]=self.circuit.solve_pos(self.vecs[0,i-1],pos1=self.plane_idxs[i-1]-1,pos2=self.plane_idxs[i]-1) 
                self.updatePlaneWFM(i)
            for i in np.arange(self.K-1)[::-1]:
                self.vecs[1,i]=self.circuit.solve_pos(self.vecs[1,i+1],pos1=self.plane_idxs[i+1],pos2=self.plane_idxs[i]) 
                self.updatePlaneWFM(i)
            
            F=np.round(self.getFidelity(),6)
            S=np.round(self.getSuccessProb(),6)

            if (abs(F-maxF)>dF and abs(S-maxS)>dS):
                maxF=F
                maxS=S
            elif((F-maxF)<dF or (S-maxS)<dS):
                stopCount+=1

            if stopCount>= patienceFid:
                break
                
        self.printVerbose('[[GenUnivUnit:WFM]] WFM convereged, Fidelity:'+str(F))       
        print('F:',self.getFidelity(),'S:',self.getSuccessProb())
        return i  #max iters
    


            