from scipy.special import gammaln
import numpy as np


def negloglike_lnp(
    w: np.array, c: np.array, s: np.array, dt: float = 0.1, R: float = 50, lam = 1.0
) -> float:
    """Implements the negative (!) log-likelihood of the LNP model

    Parameters
    ----------

    w: np.array, (Dx * Dy, )
      current receptive field

    c: np.array, (nT, )
      spike counts

    s: np.array, (Dx * Dy, nT)
      stimulus matrix


    Returns
    -------

    f: float
      function value of the negative log likelihood at w

    """

    # ------------------------------------------------
    # Implement the negative log-likelihood of the LNP
    # ------------------------------------------------

    w = w.ravel()  # ensure shape (D,)
    lin = w @ s  # shape (nT,)
    rates = np.exp(lin) * dt * R  # shape (nT,)

    # term1: ∑ c_t · (w^T s_t)
    term1 = np.dot(c, lin)

    # term2: ∑ exp(w^T s_t) · (Δt·R)
    term2 = rates.sum()

    # term3: ∑ log(c_t!) = ∑ gam    # --- Lmaln(c_t + 1)
    term3 = gammaln(c + 1).sum()

    # term4: ∑ c_t · log(Δt·R)
    log_dtR = np.log(dt * R)
    term4 = (c * log_dtR).sum()

    # full log-likelihood
    logL = term1 - term2 - term3 + term4

    reg = lam * np.sum(w**2)

    # return *negative* log-likelihood
    return -logL+ reg


def deriv_negloglike_lnp(
    w: np.array, c: np.array, s: np.array, dt: float = 0.1, R: float = 50, lam=1.0
) -> np.array:
    """
    Implements the gradient of the negative log-likelihood of the LNP model

    Parameters
    ----------

    see negloglike_lnp

    Returns
    -------

    df: np.array, (Dx * Dy, )
      gradient of the negative log likelihood with respect to w
    """
    w = w.ravel()  # ensure shape (D*D,)
    lin = w @ s  # shape (nT,)
    rates = np.exp(lin) * dt * R

    # gradient = ∑_t (r_t - c_t) s_t
    grad = s @ (rates - c)  # shape (D*D,)
    grad += 2 * lam * w
    return grad