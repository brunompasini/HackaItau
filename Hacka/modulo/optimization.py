import numpy as np

def minimize_vol(rp, e_r, cov):
    n =  e_r.shape[0]
    
    #Primeiro Chute
    w = (np.ones(n)*1/n).reshape(n, 1)#Carteira
    l = np.array([[0.],[0.]])#Multiplicadores de Lagrange
    
    #Calculando grad(c(x)) 
    dc = np.hstack((np.ones((n,1)), e_r))
    
    #Criando a Matriz A
    def generate_A(Sigma, dc, n):
        a = np.hstack([2*Sigma, dc])
        b = np.hstack([gradc.T, np.zeros((n,n))])
    return np.vstack((a, b))
    
    #Criando o vetor b.
    def generate_b(Sigma, gradc, er, l, w, rp):
        a = 2*Sigma@w + gradc@l
        b = np.ones((1,n))@w - 1
        c = er.T@w - rp
    return -1*np.vstack((a, b, c))
    
    A = generate_A(Sigma, gradc, n)
    
    for i in range(1000):
        b = generate_b(Sigma, gradc, er, l, w, rp)
        dx = np.linalg.inv(A)@b
        dw, dl = np.vsplit(dx, 2)
        w = w + 0.01*dw
        l = l + 0.01*dl

    return w