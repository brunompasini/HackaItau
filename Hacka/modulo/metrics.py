import numpy as np

def portfolio_return(w, e_r):
    """
    Essa função calcula o retorno esperado de uma carteira dado o vetor de retornos esperados.
    Args:
        w (np.array): vetor com os pesos da carteira
        e_r (np.array): vetor com os retornos esperados
    Returns:
        float: retorno esperado da carteira
    """
    return float(w.T@e_r)

def portfolio_vol(w, cov):
    """
    Essa função calcula a volatilidade de uma carteira dado a matriz de covariância.
    Args:
        w (np.array): vetor com os pesos da carteira
        cov (np.array): matriz da covariância
    Returns:
        float: volatilidade do carteira
    """
    return np.sqrt(w.T@cov@w)

def sharpe_ratio(w, e_r, cov, r_f=0):
    """
    Essa função calcula o Sharpe Ratio de uma carteira dado a matriz de covariância e vetor de retornos esperados.
    Args:
        w (np.array): vetor com os pesos da carteira
        e_r (np.array): vetor com os retornos esperados
        cov (np.array): matriz da covariância
        r_f (float): Taxa Livre de Risco
    Returns:
        float: Sharpe Ratio do portfólio
    """
    return (portfolio_return(w, e_r)-r_f)/portfolio_vol(w, cov)




class CovarianceEstimator:
    """
    Classe usada para estimar a matriz de covariância a partir da matriz estimada pela amostra e pela matriz de correlação constante.
    Baseado no artigo "Honey, I Shrunk the Sample Covariance Matrix" do Olivier Ledoit e Michael Wolf, The Journal of Portfolio Management.
    Pegamos boa parte do código da biblioteca PyPortfolioOpt.
    """
    def __init__(self, returns):
        self.returns = returns.dropna()
        self.S = returns.cov().values
    
    def honey_shrinkage(self):
        """
        Pegamos da biblioteca pyPortfolioopt.
        """
        
        
        X = np.nan_to_num(self.returns.values)
        t, n = np.shape(X)

        S = self.S  # sample cov matrix

        # Constant correlation target
        var = np.diag(S).reshape(-1, 1)
        std = np.sqrt(var)
        _var = np.tile(var, (n,))
        _std = np.tile(std, (n,))
        r_bar = (np.sum(S / (_std * _std.T)) - n) / (n * (n - 1))
        F = r_bar * (_std * _std.T)
        F[np.eye(n) == 1] = var.reshape(-1)

        # Estimate pi
        Xm = X - X.mean(axis=0)
        y = Xm ** 2
        pi_mat = np.dot(y.T, y) / t - 2 * np.dot(Xm.T, Xm) * S / t + S ** 2
        pi_hat = np.sum(pi_mat)

        # Theta matrix, expanded term by term
        term1 = np.dot((Xm ** 3).T, Xm) / t
        help_ = np.dot(Xm.T, Xm) / t
        help_diag = np.diag(help_)
        term2 = np.tile(help_diag, (n, 1)).T * S
        term3 = help_ * _var
        term4 = _var * S
        theta_mat = term1 - term2 - term3 + term4
        theta_mat[np.eye(n) == 1] = np.zeros(n)
        rho_hat = sum(np.diag(pi_mat)) + r_bar * np.sum(
            np.dot((1 / std), std.T) * theta_mat
        )

        # Estimate gamma
        gamma_hat = np.linalg.norm(S - F, "fro") ** 2

        # Compute shrinkage constant
        kappa_hat = (pi_hat - rho_hat) / gamma_hat
        delta = max(0.0, min(1.0, kappa_hat / t))

        # Compute shrunk covariance matrix
        shrunk_cov = delta * F + (1 - delta) * S
        return shrunk_cov, delta