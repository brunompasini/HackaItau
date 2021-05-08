import numpy as np
import seaborn as sns
import metrics 
import pandas as pd

class Markowitz:
    
    def __init__(self, er, cov, sim=100000):
        self.er = er
        self.cov = cov
        self.carteiras =None
        
        self.points = []
        n = er.shape[0]
        self.carteiras = []
        self.retornos = []
        self.vols = []

        for i in range(sim):
            carteira = (np.random.random(n))
            carteira = carteira/sum(carteira)
            self.carteiras.append(carteira)
            self.retornos.append(float(carteira.T@er))
            self.vols.append(float(np.sqrt(carteira.T@cov@carteira)))
        
        self.carteiras = np.array(self.carteiras)
        self.retornos = np.array(self.retornos)
        self.vols = np.array(self.vols)
        
    


    def minimize_ipm(self, rp, eps = 0.0001, alpha= 0.01):

        n =  self.er.shape[0]
        dc = np.hstack((np.ones((n,1)), self.er))
        #Primeiro Chute
        w = (np.ones(n)).reshape(n, 1)#Carteira
 
        l = -1*(np.linalg.inv(dc.T@dc)@dc.T@self.cov@w)#Multiplicadores de Lagrange
        
        #Calculando grad(c(x)) 
        #Criando a Matriz A
        def generate_A(Sigma, dc, n):
            a = np.hstack([2*Sigma, dc])
            b = np.hstack([dc.T, np.zeros((2,2))])
            return np.vstack((a, b))

        #Criando o vetor b.
        def generate_b(Sigma, dc, e_r, l, w, rp):
            a = 2*Sigma@w + dc@l
            b = np.ones((1,n))@w - 1
            c = e_r.T@w - rp
            return -1*np.vstack((a, b, c))

        def grad(w, cov, l, dc):
            return (2*cov@w+ dc@l)

        A = generate_A(self.cov, dc, n)
        i = 0
        while (abs((w.T@self.er - rp + np.sum(w) - 1 )) > 2*eps): 
            i = i+1
            b = generate_b(self.cov, dc, self.er, l, w, rp)
            dx = np.linalg.inv(A)@b
       
            dw, dl = dx[0:n][:], dx[n:n+2][:]
            w = w + alpha*dw
            
            l = l + alpha*dl
        return w
    
    def minimize_montecarlo(self, rp):
        beg = np.min(self.vols)
        end = np.max(self.vols)
        
        step = (end-beg)/1000       
        ef_x =[]
        ef_y = []
        for i in range(1000):
            x = (beg+i*step + beg+(i+1)*step)/2
            try: 
                y = np.max(self.retornos[np.logical_and((self.vols>=beg+i*step), (self.vols<=beg+(i+1)*step))])
                ef_x.append(x)
                ef_y.append(y)
            except ValueError: 
                pass
        ef_y = np.array(ef_y)
        i = np.argmin(abs(ef_y - rp))
        
       
        return self.carteiras[self.retornos == ef_y[i]]
            
       
          
    def draw_ef(self, n_points = 100):
        retornos = np.linspace(np.min(self.er), np.max(self.er), n_points)
        carteiras = [self.minimize_montecarlo(i) for i in retornos]
        
        r = [float(metrics.portfolio_return(carteira.reshape(4,1), self.er)) for carteira in carteiras]
        vol = [float(metrics.portfolio_vol(carteira.reshape(4,1), self.cov)) for carteira in carteiras]
        
        df = pd.DataFrame({"Retorno": r, "Vol":vol})
        
        return df.plot(x="Vol", y="Retorno", marker = ".", color="goldenrod", kind="scatter", figsize=(12,6))



















