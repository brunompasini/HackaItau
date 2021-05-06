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