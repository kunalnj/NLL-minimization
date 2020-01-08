import numpy as np
import copy


def U_sum(L = None, U = None, y = None, x = None, i = None, j = None, i_high = False, j_high = False, f_sub = False, b_sub = False):
    _ = []
    if i_high:
        for k in range(i):
            _.append(L[i,k]*U[k,j])
    
    if j_high:
        for k in range(j):
            _.append(L[i,k]*U[k,j])
            
    if f_sub:
        for j in range(i):
            _.append(L[i,j]*y[j])
    
    if b_sub:
        for j in range(i+1,len(x)):
            _.append(U[i,j]*x[j])
            
    return sum(_)



def LU_decomp(A):   
    
    if A.shape[0] != A.shape[1]:
        raise Exception('Entered a non-square matrix. Please enter a square matrix.')
    
    else:
        
        n = A.shape[0]
        L = np.zeros((n,n))
        U = np.zeros((n,n))
        B = np.zeros((n,n))

        for i in range(n):
            L[i,i] = 1

        for j in range(n):
            for i in range(j+1):  #upto jth index
                U[i,j] = A[i,j] - U_sum(L = L,U = U,i = i,j = j,i_high = True)
            for i in range(j+1,n):  #from j+1 index
                L[i,j] = (1/U[j,j])*(A[i,j] - U_sum(L = L,U = U,i = i,j = j,j_high = True))

        L_temp = copy.deepcopy(L)  #create a deepcopy of L so that the matrix B (matrix that the question wants the
                                   #function to return), can be calculated

        for i in range(n):
            L_temp[i,i] = 0

        B = U+L_temp  #where B is the matrix containing all elements in U and all non-diagonal elements in L
    
        return B,L,U






def det_decomp(A):
    
    n = A.shape[0]
    B = LU_decomp(A)[0]
    det_A = 1
    for i in range(n):
        det_A*=B[i,i]
    
    return det_A




def forward_back_sub(L,U,b):
    
    n = len(b)
    
    y = np.zeros((n,1))
    y[0] = b[0]/L[0,0]
    
    for i in range(1,n):
        y[i] = (1/L[i,i])*(b[i] - U_sum(L = L, y = y,i = i, f_sub = True))
    
    x = np.zeros((n,1))
    x[n-1] = y[n-1]/U[n-1,n-1]
    
    for i in range(n-2,-1,-1):
        x[i] = (1/U[i,i])*(y[i] - U_sum(U = U,x = x, i = i, b_sub = True))
        
    return x.ravel()




def inverse(A):
    n = A.shape[0]
    A_inv = np.zeros((n,n))
    
    for i in range(n):
        b = np.zeros((n,1))
        b[i] = 1
        for j in range(n):
            A_inv[j,i] = forward_back_sub(L = LU_decomp(A)[1], U = LU_decomp(A)[2], b = b)[j]
    
    return A_inv  





