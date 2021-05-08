import numpy as np
import seaborn as sns
import metrics 
import pandas as pd

class Markowitz:
    """
    Classe que implmenta o problema de Markowitz de duas formas.
    
    -Usando o método do ponto interior: http://cepac.cheme.cmu.edu/pasilectures/biegler/ipopt.pdf
     Não funciona tão bem. 
    
    - Gerando diversos portfólios randomicos e escolhendo aquele que melhor se aproxima do retorno desejados
    """
    def __init__(self, er, cov, sim=100000):
        """
        Salva a matriz de covariãncia e o vetor de retornos esperados. 
        Gera dezenas de carteiras aleatórias.
        """
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
        """
        Resolve o Problema de Markowitz usando o método do ponto interior.
        """

        def generate_b(Sigma, gradc, er, l, w, rp):
            n = er.shape[0]
            a = 2*Sigma@w + gradc@l
            b = np.ones((1,n))@w - 1
            c = er.T@w - rp
            return -1*np.vstack((a, b, c))

        def generate_A(Sigma,gradc):
            a = np.hstack([2*Sigma, gradc])
            b = np.hstack([gradc.T, np.zeros((2,2))])
            return np.vstack((a, b))


        n = self.er.shape[0]
        w = np.ones((n,1))*1/n

        dc = np.hstack((np.ones((n,1)), self.er))

        l = -1*(np.linalg.inv(dc.T@dc)@dc.T@self.cov@w)
        A = generate_A(self.cov, dc)

        for i in range(1000):
            b = generate_b(self.cov, dc, self.er, l, w, rp)
            dx = np.linalg.inv(A)@b
            dw = dx[:n][:]
            dl = dx[n+1:][:]
            w = w + 0.01*dw
            l = l + 0.01*dl
            
        return w
    
    def minimize_montecarlo(self, rp):
        """
        Minimiza pelo método de MonteCarlo.
        """
        
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
            
       
          
    def draw_ef(self, n_points = 100, method = "montecarlo"):
        """
        Retorna um Plot da fronteira eficiente + um DataFrame com retornos, volatilidade e carteiras.
        """
        
        if method == "montecarlo":
            method = self.minimize_montecarlo
            
        else:
            method = self.minimize_ipm
        retornos = np.linspace(np.min(self.er), np.max(self.er), n_points)
        carteiras = [method(i).reshape(self.er.shape[0]) for i in retornos]
        
        r = [float(metrics.portfolio_return(carteira.reshape(4,1), self.er)) for carteira in carteiras]
        vol = [float(metrics.portfolio_vol(carteira.reshape(4,1), self.cov)) for carteira in carteiras]
        
        df = pd.DataFrame({"Retorno": r, "Vol":vol})
        df  = pd.concat([df, pd.DataFrame(carteiras)], axis = 1)
        
       
        return df.plot(x="Vol", y="Retorno", marker = ".", color="goldenrod", kind="scatter", figsize=(12,6)), df 



















