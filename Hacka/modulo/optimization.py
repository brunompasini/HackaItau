import numpy as np

def minimize_vol(rp, e_r, cov, eps = 0.0001, alpha= 0.01):
    n =  e_r.shape[0]
    dc = np.hstack((np.ones((n,1)), e_r))
    #Primeiro Chute
    w = (np.ones(n)*1/n).reshape(n, 1)#Carteira
    w = w/sum(w)
    l = -1*(np.linalg.inv(dc.T@dc)@dc.T@cov@w)#Multiplicadores de Lagrange
    
    #Calculando grad(c(x)) 
    #Criando a Matriz A
    def generate_A(Sigma, dc, n, delta1 = 0, delta2 =0.0):
        a = np.hstack([2*Sigma+delta1*np.eye(len(w)), dc])
        b = np.hstack([dc.T, np.zeros((2,2))-delta2*np.eye(2)])
        return np.vstack((a, b))
    
    #Criando o vetor b.
    def generate_b(Sigma, dc, e_r, l, w, rp):
        a = 2*Sigma@w + dc@l
        b = np.ones((1,n))@w - 1
        c = e_r.T@w - rp
        return -1*np.vstack((a, b, c))
    
    def grad(w, cov, l, dc):
        return (2*cov@w+ dc@l)
    
    A = generate_A(cov, dc, n)
    
    while np.sum(grad(w, cov, l, dc)) > 2*eps:                                               #while max(np.abs((grad(w,cov, l, dc)))) > eps:
        b = generate_b(cov, dc, e_r, l, w, rp)
        #print("b: ",b, "\n")
        dx = np.linalg.inv(A)@b
        #print("dx: ", dx, "\n")
        dw, dl = dx[0:n][:], dx[n:n+2][:]
        w = w + alpha*dw
        #print("w: \n", w, "\n")
        l = l + alpha*dl
        
    return w






















