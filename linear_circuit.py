
from scipy.stats import unitary_group
from scipy.linalg import dft
import numpy as np
class LinearCircuit():
    """Linear Circuit
    Circuit design for consitent matrix multiplications across a long circuit
    """
    def __init__(self,verbose=0):
        self.C=[]
        self.shape=(0,0)
        self.verbose=verbose
        self.printVerbose('[[Circuit]] Initilized a new circuit')
    
    def updateShape(self):  
        """
        Updates shape of the circuit after all internal matricies are multiplied
        """
        if len(self.C) ==0:
                self.shape=(0,0)
        else :
                self.shape= (self.C[-1].shape[0],self.C[0].shape[1])      
                
    def printVerbose(self,message):
        """
        for Verbose messages
        """
        if self.verbose>0:
            print(message)
                    
    def checkPos(self,i,min=1,max=None):
        """
        Checks if a given position i is in the circuit
        """
        if max is None:
            max=self.getCirLen()+1
        if i not in range(min,max):
            raise(Exception('Position should be between 1 and '+str(self.getCirLen())))
        
    def add(self,U):   
        """Add layer to a circuit
        """
        U=np.matrix(U)

        if len(self.C) ==0:
            self.C.append(U)
            self.printVerbose('[[Circuit:Add]] Added new matrix:'+str(U))
        else:
            if self.shape[0] == U.shape[1]:
                self.C.append(U)
                self.printVerbose('[[Circuit:Add]] Added new matrix'+str(U))
            else:
                raise(Exception('Shape of the supplied operation '+str(self.shape)+' is not appendable to the circuit '+str(U.shape)))
        self.updateShape()

        
                
    def getLayer(self,i):  
        """get ith layer of circuit
        """
        self.checkPos(i)
        return self.C[i-1]
    
    def getCirLen(self):  
        """get length of circuit
        """
        return len(self.C)
    
    def updateLayer(self,Vm,i):  
        """update layer in a circuit
        
        Changes a matrix in the cirucit
        """
        V=np.matrix(Vm)
        self.checkPos(i)
            
        condR=(V.shape[0]==self.getLayer(i).shape[0]) #condition for same Rows
        condC=(V.shape[1]==self.getLayer(i).shape[1]) #condition for same columns
        
        if (condR and condC) or (i==1 and condR) or (i==self.getCirLen() and condC):
            self.C[i-1]=V
            self.printVerbose('[[Circuit:UpdateLayer]] Updated Layer at position'+str(i))
        else:
            raise(Exception('Dimension mismatch between new'+str(V.shape)+' and previous operation '+str(self.getLayer(i).shape)+'at position'+str(i)))
 
    
    def print(self,precision=1,supp_small=True):  #print circuit
        """
        Prints the matricies involved in the circuit
        """
        for C in self.C:
            print(np.array_str(C,precision=precision,suppress_small=supp_small ))
            print('=>=>=>')
            
            
    def solve(self,x,dir,pos,use_inverse_for_bkwd=False):  
        """Solve circuit in forward or backward direction till a certain position
        
        dir can be 1 or -1, x is the input array
        """
        if dir not in [1,-1]:
            raise(Exception('Direction can only be -1 or 1'))
        #lenX=x.shape[1]
        x=np.matrix(np.array(x)).T #Conv to a column matrix
        self.checkPos(pos,min=0)
        dir01=np.int((dir+1)/2)  #conv dir 1,-1 to 0,1
            
        nx=x[:]
        if dir==1:
            for V in self.C [:pos]:
                nx=V@nx
        else:
            for V in self.C[::-1][:self.getCirLen()-pos]:
                if use_inverse_for_bkwd:
                    nx=np.linalg.inv(V)@nx
                else:
                    nx=(V.conj().T)@nx
        self.printVerbose('[[Circuit:Solve]] Solved for x at position'+str(pos)+'in direction '+str(dir))
        return np.array(nx.T).squeeze()
      
    def solve_pos(self,x,pos1,pos2,use_inverse_for_bkwd=False):  
        """Solve circuit from given position to other position till a certain position for a given array of input vectors
        """

        lenX=len(x)
        x=np.matrix(np.array(x)).T #Conv to a column matrix
        self.checkPos(pos1,min=0)
        self.checkPos(pos2,min=0)
        nx=x[:]
        if pos2>pos1:
            for V in self.C [pos1:pos2]:
                nx=V@nx
        else:
            for V in self.C[pos2:pos1][::-1]:
                if use_inverse_for_bkwd:
                    nx=np.linalg.inv(V)@nx
                else:
                    nx=(V.conj().T)@nx
        self.printVerbose('[[Circuit:Solve]] Solved for x from position'+str(pos1)+'to position '+str(pos2))
        return np.array(nx.T).squeeze()#[0]
      