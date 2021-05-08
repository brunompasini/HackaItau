import numpy as np

def portfolio_return(w : np.ndarray, e_r : np.ndarray) -> float:
    """Essa função calcula o retorno esperado de uma carteira dado o vetor de retornos esperados.
    Args:
        w (np.ndarray): vetor com os pesos da carteira
        e_r (np.ndarray): vetor com os retornos esperados
    Returns:
        float: retorno esperado da carteira
    """
    return float(w.T@e_r)

def portfolio_vol(w : np.ndarray, cov : np.ndarray) -> float:
    """Essa função calcula a volatilidade de uma carteira dado a matriz de covariância.
    Args:
        w (np.ndarray): vetor com os pesos da carteira
        cov (np.ndarray): matriz da covariância
    Returns:
        float: volatilidade do carteira
    """
    return np.sqrt(w.T@cov@w)

def sharpe_ratio(w : np.ndarray, e_r : np.ndarray,
                 cov : np.ndarray, r_f : float = 0) -> float:
    """Essa função calcula o Sharpe Ratio de uma carteira dado a matriz de covariância e vetor de retornos esperados.
    Args:
        w (np.ndarray): vetor com os pesos da carteira
        e_r (np.ndarray): vetor com os retornos esperados
        cov (np.ndarray): matriz da covariância
        r_f (float): Taxa Livre de Risco
    Returns:
        float: Sharpe Ratio do portfólio
    """
    return (portfolio_return(w, e_r)-r_f)/portfolio_vol(w, cov)
   